import re
import logging
from typing import List, Optional

from models import ExtractionResult, SchedulePageResult, ScheduleRow
from pdf_processor import open_document

logger = logging.getLogger(__name__)


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
        try:
            tables_finder = page.find_tables()
        except Exception:
            tables_finder = None
        found_tables = tables_finder.tables if tables_finder else []
        for table in found_tables:
            try:
                if getattr(table, "row_count", 0) <= 2:
                    continue

                grid_data = table.extract()
                if not grid_data or not grid_data[0]:
                    continue

                lower_header = [str(h or "").lower().strip() for h in grid_data[0]]
                try:
                    identity_idx = next(i for i, h in enumerate(lower_header) if "identity" in h)
                    desc_idx = next(i for i, h in enumerate(lower_header) if "description" in h)
                    cost_idx = next(i for i, h in enumerate(lower_header) if "cost" in h)
                    val_idx = next(i for i, h in enumerate(lower_header) if h.endswith("value"))
                except StopIteration:
                    continue

                filtered_rows: List[List[str]] = []
                for row in grid_data[1:]:
                    cost_val = re.sub(r"cost\s*\*\*", "", (row[cost_idx] or ""), flags=re.IGNORECASE).strip()
                    if cost_val == "**":
                        cost_val = ""
                    new_row = [row[identity_idx] or "", row[desc_idx] or "", cost_val, row[val_idx] or ""]
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
    for i in page_indices:
        if i < 0 or i >= len(doc):
            continue
        rows = extract_grid_rows_from_page(doc[i], i)
        if rows:
            return True
    return False


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
            rows = extract_grid_rows_from_page(doc[i], i)
            schedule_rows: List[ScheduleRow] = []
            for r in rows:
                identity, desc, cost, value = (r + [""] * 4)[:4]
                schedule_rows.append(ScheduleRow(identity=identity or "", description=desc or "", cost=cost or None, current_value=value or ""))

            page_result = SchedulePageResult(page_number=i + 1, rows=schedule_rows, has_cost_column=any(r.cost for r in schedule_rows), source="TEXT")
            page_results.append(page_result)

        return ExtractionResult(plan_name=plan_name, pages=page_results)
    finally:
        doc.close()
