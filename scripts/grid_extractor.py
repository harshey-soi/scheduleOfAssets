import re
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models import ExtractionResult, SchedulePageResult, ScheduleRow
from pdf_processor import extract_words, open_document, render_page_to_image
from schedule_parser import parse_schedule_h_page

logger = logging.getLogger(__name__)


def _count_clustered_positions(indices: np.ndarray, max_gap: int = 3) -> int:
    if indices.size == 0:
        return 0

    count = 1
    last = int(indices[0])
    for idx in indices[1:]:
        current = int(idx)
        if current - last > max_gap:
            count += 1
        last = current
    return count


def _count_intersection_clusters(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clusters = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 4:
            clusters += 1
    return clusters


def _get_image_grid_structure_metrics(page) -> Tuple[bool, int, int, int]:
    try:
        image = render_page_to_image(page, zoom=2.0)
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        height, width = binary.shape[:2]
        top = int(height * 0.22)
        bottom = int(height * 0.97)
        region = binary[top:bottom, :]
        if region.size == 0:
            return False, 0, 0, 0

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 18), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, region.shape[0] // 18)))
        horizontal = cv2.morphologyEx(region, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(region, cv2.MORPH_OPEN, vertical_kernel)

        horizontal_projection = np.count_nonzero(horizontal, axis=1)
        vertical_projection = np.count_nonzero(vertical, axis=0)

        horizontal_indices = np.where(horizontal_projection >= int(width * 0.35))[0]
        vertical_indices = np.where(vertical_projection >= int(region.shape[0] * 0.22))[0]

        horizontal_count = _count_clustered_positions(horizontal_indices)
        vertical_count = _count_clustered_positions(vertical_indices)

        intersections = cv2.bitwise_and(horizontal, vertical)
        intersection_count = _count_intersection_clusters(intersections)

        logger.debug(
            "Grid image check: horiz=%d vert=%d intersections=%d",
            horizontal_count,
            vertical_count,
            intersection_count,
        )

        is_grid = horizontal_count >= 8 and vertical_count >= 4 and intersection_count >= 12
        return is_grid, horizontal_count, vertical_count, intersection_count
    except Exception as exc:
        logger.debug("Grid image check failed: %s", exc)
        return False, 0, 0, 0


def _to_point_tuple(point) -> Optional[Tuple[float, float]]:
    try:
        if hasattr(point, "x") and hasattr(point, "y"):
            return float(point.x), float(point.y)
        if isinstance(point, (tuple, list)) and len(point) >= 2:
            return float(point[0]), float(point[1])
    except Exception:
        return None
    return None


def _to_rect_tuple(rect) -> Optional[Tuple[float, float, float, float]]:
    try:
        if rect is None:
            return None
        if all(hasattr(rect, attr) for attr in ("x0", "y0", "x1", "y1")):
            return float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
        if isinstance(rect, (tuple, list)) and len(rect) >= 4:
            return float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    except Exception:
        return None
    return None


def _expand_rect(rect: Tuple[float, float, float, float], margin: float) -> Tuple[float, float, float, float]:
    return rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin


def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _segment_intersects_rect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    rect: Tuple[float, float, float, float],
) -> bool:
    x_min = min(p1[0], p2[0])
    x_max = max(p1[0], p2[0])
    y_min = min(p1[1], p2[1])
    y_max = max(p1[1], p2[1])
    seg_rect = (x_min, y_min, x_max, y_max)
    return _rects_overlap(seg_rect, rect)


def _has_visual_grid_structure(page, table_bbox) -> bool:
    ok, _, _, _ = _get_visual_grid_structure_metrics(page, table_bbox)
    return ok


def _get_visual_grid_structure_metrics(page, table_bbox) -> Tuple[bool, int, int, int]:
    bbox = _to_rect_tuple(table_bbox)
    if bbox is None:
        return False, 0, 0, 0

    probe_rect = _expand_rect(bbox, 6.0)
    bbox_width = max(1.0, bbox[2] - bbox[0])
    bbox_height = max(1.0, bbox[3] - bbox[1])
    min_horizontal_span = max(20.0, bbox_width * 0.15)
    min_vertical_span = max(12.0, bbox_height * 0.15)

    horizontal_positions = set()
    vertical_positions = set()
    rectangle_hits = 0

    try:
        drawings = page.get_drawings() or []
    except Exception:
        drawings = []

    for drawing in drawings:
        drawing_rect = _to_rect_tuple(drawing.get("rect"))
        if drawing_rect and not _rects_overlap(probe_rect, _expand_rect(drawing_rect, 2.0)):
            continue

        for item in drawing.get("items", []):
            try:
                kind = item[0]
            except Exception:
                continue

            if kind == "l" and len(item) >= 3:
                p1 = _to_point_tuple(item[1])
                p2 = _to_point_tuple(item[2])
                if p1 is None or p2 is None:
                    continue
                if not _segment_intersects_rect(p1, p2, probe_rect):
                    continue

                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])
                if dx >= min_horizontal_span and dy <= 1.5:
                    horizontal_positions.add(round((p1[1] + p2[1]) / 2.0, 1))
                elif dy >= min_vertical_span and dx <= 1.5:
                    vertical_positions.add(round((p1[0] + p2[0]) / 2.0, 1))

            elif kind == "re" and len(item) >= 2:
                rect = _to_rect_tuple(item[1])
                if rect and _rects_overlap(probe_rect, rect):
                    rect_width = abs(rect[2] - rect[0])
                    rect_height = abs(rect[3] - rect[1])
                    if rect_width >= min_horizontal_span and rect_height >= min_vertical_span:
                        rectangle_hits += 1
                        horizontal_positions.add(round(rect[1], 1))
                        horizontal_positions.add(round(rect[3], 1))
                        vertical_positions.add(round(rect[0], 1))
                        vertical_positions.add(round(rect[2], 1))

    logger.debug(
        "Grid visual check: bbox=%s horiz_positions=%d vert_positions=%d rects=%d",
        bbox,
        len(horizontal_positions),
        len(vertical_positions),
        rectangle_hits,
    )
    is_grid = len(horizontal_positions) >= 3 and len(vertical_positions) >= 2
    return is_grid, len(horizontal_positions), len(vertical_positions), rectangle_hits


def extract_grid_rows_from_page(page, page_index: int) -> List[List[str]]:
    """Extract grid-format SOA rows from a page with orientation fallbacks."""
    try:
        logger.info("Analyzing page %d for Grid SOA.", page_index + 1)
        original_rotation = getattr(page, "rotation", 0)
        attempt_rotations = [original_rotation, (original_rotation + 90) % 360, (original_rotation + 270) % 360]

        best_rows: List[List[str]] = []
        best_rotation = original_rotation
        for rotation in attempt_rotations:
            try:
                page.set_rotation(rotation)
            except Exception:
                pass
            rows = _extract_grid_rows_current_orientation(page)
            if len(rows) > len(best_rows):
                best_rows = rows
                best_rotation = rotation

        try:
            page.set_rotation(best_rotation)
        except Exception:
            pass
        if best_rows and best_rotation != original_rotation:
            logger.info("Used rotation %d for better grid extraction on page %d.", best_rotation, page_index + 1)
        return best_rows
    except Exception as error:
        logger.warning("Grid detection failed for page %d: %s", page_index + 1, error)
        return []


def _extract_grid_rows_current_orientation(page) -> List[List[str]]:
    try:
        tables_finder = None
        image_grid_metrics = None
        try:
            tables_finder = page.find_tables()
        except Exception:
            tables_finder = None
        found_tables = tables_finder.tables if tables_finder else []
        if not found_tables:
            logger.debug("Grid extractor: page has no tables from find_tables()")
        for table in found_tables:
            try:
                if getattr(table, "row_count", 0) <= 2:
                    logger.debug(
                        "Grid extractor: skipping detected table with too few rows (row_count=%s)",
                        getattr(table, "row_count", 0),
                    )
                    continue

                grid_data = table.extract()
                if not grid_data or not grid_data[0]:
                    logger.debug("Grid extractor: skipping detected table with empty extracted header row")
                    continue

                header_row_idx = None
                lower_header = []
                identity_idx = desc_idx = cost_idx = val_idx = None
                header_scan_limit = min(6, len(grid_data))
                header_samples = []
                for candidate_idx in range(header_scan_limit):
                    candidate_header = [str(h or "").lower().strip() for h in grid_data[candidate_idx]]
                    header_samples.append(candidate_header)
                    try:
                        candidate_identity_idx = next(i for i, h in enumerate(candidate_header) if "identity" in h)
                        candidate_desc_idx = next(i for i, h in enumerate(candidate_header) if "description" in h)
                        candidate_cost_idx = next(i for i, h in enumerate(candidate_header) if "cost" in h)

                        val_candidates = [
                            i for i, h in enumerate(candidate_header) if "current" in h or "current value" in h
                        ]
                        if not val_candidates:
                            val_candidates = [i for i, h in enumerate(candidate_header) if "value" in h and i != candidate_desc_idx]

                        if val_candidates:
                            candidate_val_idx = val_candidates[-1]
                        else:
                            candidate_val_idx = next(
                                (
                                    i for i in range(len(candidate_header) - 1, -1, -1)
                                    if i not in (candidate_identity_idx, candidate_desc_idx, candidate_cost_idx)
                                ),
                                None,
                            )
                        if candidate_val_idx is None:
                            raise StopIteration

                        header_row_idx = candidate_idx
                        lower_header = candidate_header
                        identity_idx = candidate_identity_idx
                        desc_idx = candidate_desc_idx
                        cost_idx = candidate_cost_idx
                        val_idx = candidate_val_idx
                        break
                    except StopIteration:
                        continue

                if header_row_idx is None or identity_idx is None or desc_idx is None or cost_idx is None or val_idx is None:
                    logger.debug(
                        "Grid extractor: table header rejected; scanned first %d rows, samples=%s",
                        header_scan_limit,
                        header_samples,
                    )
                    continue

                has_visual_grid, horiz_count, vert_count, rect_count = _get_visual_grid_structure_metrics(
                    page,
                    getattr(table, "bbox", None),
                )
                if not has_visual_grid:
                    if image_grid_metrics is None:
                        image_grid_metrics = _get_image_grid_structure_metrics(page)
                    image_grid_ok, image_horiz, image_vert, image_intersections = image_grid_metrics
                    if image_grid_ok:
                        logger.debug(
                            "Grid extractor: accepting table-like content on page via image grid detector; header=%s image_horiz=%d image_vert=%d image_intersections=%d",
                            lower_header,
                            image_horiz,
                            image_vert,
                            image_intersections,
                        )
                    else:
                        logger.debug(
                            "Grid extractor: table-like content on page rejected by visual grid check; header=%s horiz_positions=%d vert_positions=%d rects=%d bbox=%s image_horiz=%d image_vert=%d image_intersections=%d",
                            lower_header,
                            horiz_count,
                            vert_count,
                            rect_count,
                            _to_rect_tuple(getattr(table, "bbox", None)),
                            image_horiz,
                            image_vert,
                            image_intersections,
                        )
                        continue

                filtered_rows: List[List[str]] = []
                # Helper to detect numeric-like cells (amounts)
                numeric_re = re.compile(r"[0-9][0-9,\. ]*")

                for row in grid_data[header_row_idx + 1 :]:
                    cost_val = re.sub(r"cost\s*\*\*", "", (row[cost_idx] or ""), flags=re.IGNORECASE).strip()
                    if cost_val == "**":
                        cost_val = ""

                    identity_val = row[identity_idx] or ""
                    desc_val = row[desc_idx] or ""
                    value_val = row[val_idx] or ""

                    # If the description cell itself contains a trailing numeric value
                    # (common when the extractor merges adjacent cells), split it out.
                    try:
                        m = re.search(r"^(.*?)[\s\u00A0]*([\(\$]?\d[0-9,\.\s\)\-:]*)$", desc_val.strip())
                        if m:
                            possible_desc = m.group(1).strip()
                            possible_value = m.group(2).strip().strip(',:;')
                            # Only treat as a split if the trailing part looks numeric
                            if possible_value and numeric_re.search(possible_value):
                                logger.debug(
                                    "Grid extractor: splitting trailing value from description for identity=%r -> desc=%r value=%r",
                                    identity_val,
                                    possible_desc,
                                    possible_value,
                                )
                                # prefer an explicit value cell if already present, otherwise use split
                                if not value_val or not numeric_re.search(value_val):
                                    value_val = possible_value
                                desc_val = possible_desc
                    except Exception:
                        pass

                    # If the extracted value looks wrong (empty or duplicates the description),
                    # attempt to find a numeric-looking token in the remaining cells as the value.
                    if (not value_val or value_val.strip() == desc_val.strip() or not numeric_re.search(value_val)):
                        found = None
                        # Search all cells (except identity/description/cost) for numeric-like content
                        for j in range(len(row)):
                            if j in (identity_idx, desc_idx, cost_idx):
                                continue
                            cell = (row[j] or "").strip()
                            if not cell:
                                continue
                            if numeric_re.search(cell):
                                # collect numeric fragments to the left
                                parts_left = []
                                k = j - 1
                                while k >= 0:
                                    if k in (identity_idx, desc_idx, cost_idx):
                                        k -= 1
                                        continue
                                    prev = (row[k] or "").strip()
                                    if prev and numeric_re.search(prev):
                                        parts_left.insert(0, prev)
                                        k -= 1
                                    else:
                                        break

                                # collect numeric fragments to the right
                                parts_right = [cell]
                                k = j + 1
                                while k < len(row):
                                    if k in (identity_idx, desc_idx, cost_idx):
                                        k += 1
                                        continue
                                    nxt = (row[k] or "").strip()
                                    if nxt and numeric_re.search(nxt):
                                        parts_right.append(nxt)
                                        k += 1
                                    else:
                                        break

                                parts = parts_left + parts_right
                                candidate = "".join(p.replace(" ", "") for p in parts)
                                candidate = candidate.strip().strip(',:;')
                                if candidate:
                                    found = candidate
                                    break

                        # If no numeric fragments found, fall back to last non-empty cell
                        if not found:
                            last_nonempty = None
                            for j in range(len(row) - 1, -1, -1):
                                if j in (identity_idx, desc_idx):
                                    continue
                                cell = (row[j] or "").strip()
                                if cell:
                                    last_nonempty = cell
                                    break
                            if last_nonempty and last_nonempty.strip() != desc_val.strip():
                                found = last_nonempty.strip()

                        # If still no candidate found, emit diagnostics so we can see
                        # the raw table structure that caused the failure.
                        if not found:
                            try:
                                logger.warning(
                                    "Grid extractor: unable to find numeric value for row on page (identity=%r). header=%s row=%s",
                                    identity_val,
                                    lower_header,
                                    [str(c or "") for c in row],
                                )
                            except Exception:
                                logger.warning("Grid extractor: unable to find numeric value for row; failed to log details")

                        if found:
                            logger.debug(
                                "Grid extractor: corrected value for row (identity=%r) from %r to %r",
                                identity_val,
                                value_val,
                                found,
                            )
                            value_val = found

                    new_row = [identity_val, desc_val, cost_val, value_val]
                    filtered_rows.append(new_row)

                if filtered_rows:
                    logger.info("Identified and extracted Grid SOA with %d rows.", len(filtered_rows))

                    return filtered_rows
            except Exception:
                continue
    except Exception:
        pass
    return []


def is_grid_among_pages(doc, page_indices: List[int]) -> bool:
    return bool(get_grid_page_indices(doc, page_indices))


def get_grid_page_indices(doc, page_indices: List[int]) -> List[int]:
    grid_pages: List[int] = []
    for i in page_indices:
        if i < 0 or i >= len(doc):
            continue
        rows = extract_grid_rows_from_page(doc[i], i)
        if rows:
            grid_pages.append(i)
            continue

        image_grid_ok, image_horiz, image_vert, image_intersections = _get_image_grid_structure_metrics(doc[i])
        if image_grid_ok:
            logger.info(
                "Grid classifier: page %d accepted via image grid detector (horiz=%d vert=%d intersections=%d)",
                i + 1,
                image_horiz,
                image_vert,
                image_intersections,
            )
            grid_pages.append(i)
        else:
            logger.debug(
                "Grid classifier: page %d rejected by image grid detector (horiz=%d vert=%d intersections=%d)",
                i + 1,
                image_horiz,
                image_vert,
                image_intersections,
            )
    return grid_pages


def process_grid_pdf(pdf_path: str, page_indices: Optional[List[int]] = None) -> ExtractionResult:
    doc = open_document(pdf_path)
    try:
        if page_indices is None:
            page_indices = list(range(len(doc)))

        plan_name = "Unknown Plan Name"
        if doc.page_count > 0:
            text = doc[0].get_text("text") or ""
            for line in text.splitlines():
                line = line.strip()
                if line:
                    plan_name = line
                    break

        page_results: List[SchedulePageResult] = []
        for i in page_indices:
            page = doc[i]
            rows = extract_grid_rows_from_page(page, i)

            if not rows:
                logger.info("Grid extractor: using text fallback on grid-classified page %d", i + 1)
                words, source = extract_words(page)
                parsed_rows, has_cost_column = parse_schedule_h_page(words)
                if not parsed_rows:
                    logger.info("Grid extractor: no rows recovered from text fallback on page %d", i + 1)
                    continue

                page_result = SchedulePageResult(
                    page_number=i + 1,
                    rows=parsed_rows,
                    has_cost_column=has_cost_column,
                    source="GRID",
                )
                page_results.append(page_result)
                continue

            schedule_rows: List[ScheduleRow] = []
            for r in rows:
                identity, desc, cost, value = (r + [""] * 4)[:4]
                schedule_rows.append(ScheduleRow(identity=identity or "", description=desc or "", cost=cost or None, current_value=value or ""))

            page_result = SchedulePageResult(page_number=i + 1, rows=schedule_rows, has_cost_column=any(r.cost for r in schedule_rows), source="GRID")
            page_results.append(page_result)

        return ExtractionResult(plan_name=plan_name, pages=page_results)
    finally:
        doc.close()
