"""PDF-level operations: opening documents, locating Schedule H pages,
normalizing page orientation, and producing a source-agnostic stream of
`Word` objects for the parser to consume.

The rest of the pipeline (schedule_parser.py) never touches PyMuPDF,
pytesseract, or raw text lines directly -- everything it receives has
already been normalized into `Word` objects with PDF-point coordinates,
regardless of whether the page was natively searchable or had to be OCR'd.
"""
from __future__ import annotations

import logging
import re
import statistics
import unicodedata

import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import config as config_module

from config import (
    DESCRIPTION_HEADER_RE,
    DESKEW_MIN_ANGLE_DEG,
    IDENTITY_HEADER_RE,
    MIN_SEARCHABLE_TEXT_CHARS,
    OCR_TESSERACT_CONFIG,
    OCR_ZOOM,
    QUICK_OCR_ZOOM,
    SCHEDULE_H_HEADING_RE,
    VALUE_HEADER_RE,
)
from ocr_preprocessing import deskew_image_with_angle, preprocess_for_ocr
from models import PlanMetadata
from utils import sanitize_filename

logger = logging.getLogger(__name__)

SCHEDULE_OF_ASSETS_FALLBACK_RE = getattr(
    config_module,
    "SCHEDULE_OF_ASSETS_FALLBACK_RE",
    re.compile(r"schedule\s+of\s+assets\b", re.IGNORECASE),
)


def open_document(pdf_path: str) -> fitz.Document:
    """Open a PDF file, raising a clear error if it cannot be opened."""
    try:
        return fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - single clear failure boundary
        raise IOError(f"Could not open PDF '{pdf_path}': {exc}") from exc


def render_page_to_image(page: "fitz.Page", zoom: float) -> Image.Image:
    """Rasterize a page (respecting its currently-set rotation) to a PIL image."""
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
    if mode == "RGBA":
        image = image.convert("RGB")
    return image


def _page_text_or_quick_ocr(doc: "fitz.Document", page_index: int) -> str:
    """Return page text, falling back to a fast, low-resolution OCR pass
    when the page has little or no extractable native text.

    This is intentionally cheap: it is used to *scan* every page of a
    (potentially very long) filing looking for Schedule H, so it must stay
    fast across hundreds of PDFs. Full-quality OCR is reserved for the
    handful of pages actually identified as Schedule H content.
    """
    page = doc[page_index]
    text = page.get_text("text") or ""
    if len(text.strip()) >= MIN_SEARCHABLE_TEXT_CHARS:
        return text

    # Fall back to a quick OCR pass. Try multiple rotations and pick the
    # OCR result with the most non-whitespace characters to handle rotated
    # scanned pages more robustly.
    try:
        image = render_page_to_image(page, zoom=QUICK_OCR_ZOOM)
        attempts = [image, image.rotate(90, expand=True), image.rotate(270, expand=True)]

        best_text = ""
        for attempt in attempts:
            try:
                ocr_text = pytesseract.image_to_string(attempt, config=OCR_TESSERACT_CONFIG) or ""
            except Exception as exc_inner:  # noqa: BLE001
                logger.debug("Quick OCR rotation attempt failed on page %d: %s", page_index + 1, exc_inner)
                ocr_text = ""

            if len(ocr_text.strip()) > len(best_text.strip()):
                best_text = ocr_text

        if best_text:
            return best_text
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quick OCR probe failed on page %d: %s", page_index + 1, exc)
        return text


def extract_plan_name(doc: "fitz.Document") -> str:
    """Extract the plan name from the first non-blank line of page 1."""
    if doc.page_count == 0:
        return "Unknown Plan Name"

    text = _page_text_or_quick_ocr(doc, 0)
    for line in text.splitlines():
        line = line.strip()
        if line:
            return sanitize_filename(line)
    return "Unknown Plan Name"


def _build_visual_text_lines(words) -> list[str]:
    """Reconstruct visual rows so form labels stay beside their values."""
    boxes = [word for word in words if len(word) >= 5 and str(word[4]).strip()]
    if not boxes:
        return []

    heights = [float(word[3]) - float(word[1]) for word in boxes if word[3] > word[1]]
    tolerance = max(2.0, statistics.median(heights) * 0.6 if heights else 2.0)
    grouped_rows: list[list] = []
    row_centers: list[float] = []

    for word in sorted(boxes, key=lambda item: ((float(item[1]) + float(item[3])) / 2, float(item[0]))):
        center = (float(word[1]) + float(word[3])) / 2
        if grouped_rows and abs(center - row_centers[-1]) <= tolerance:
            grouped_rows[-1].append(word)
            row_centers[-1] = sum(
                (float(item[1]) + float(item[3])) / 2 for item in grouped_rows[-1]
            ) / len(grouped_rows[-1])
        else:
            grouped_rows.append([word])
            row_centers.append(center)

    return [
        " ".join(str(word[4]).strip() for word in sorted(row, key=lambda item: float(item[0])))
        for row in grouped_rows
    ]


def _extract_schedule_header_plan_name(lines: list[str]) -> str:
    """Extract the plan name printed directly above a Schedule H heading."""
    heading_pattern = re.compile(
        r"form\s+5500.*schedule\s+h|schedule\s+of\s+assets",
        flags=re.IGNORECASE,
    )
    generic_line_pattern = re.compile(
        r"^(?:form\s+5500|schedule\s+h|schedule\s+of\s+assets|"
        r"held\s+at\s+end\s+of\s+year)",
        flags=re.IGNORECASE,
    )

    for heading_index, line in enumerate(lines):
        if not heading_pattern.search(line):
            continue
        window_start = max(0, heading_index - 6)
        preceding_lines = lines[window_start:heading_index]
        for relative_index in range(len(preceding_lines) - 1, -1, -1):
            candidate_line = preceding_lines[relative_index].strip()
            if generic_line_pattern.search(candidate_line):
                continue
            if not re.search(r"\bplan\b", candidate_line, flags=re.IGNORECASE):
                continue

            name_parts = [candidate_line]
            if relative_index > 0 and (
                candidate_line.startswith(")")
                or re.match(r"^[A-Za-z]\)", candidate_line)
                or preceding_lines[relative_index - 1].rstrip().endswith("(")
            ):
                name_parts.insert(0, preceding_lines[relative_index - 1].strip())

            for continuation in preceding_lines[relative_index + 1:]:
                if re.match(r"^(?:for|of|and|the)\b", continuation.strip(), re.IGNORECASE):
                    name_parts.append(continuation.strip())
                else:
                    break

            plan_name = re.sub(r"\s+", " ", " ".join(name_parts)).strip(" :#-\t")
            plan_name = re.sub(r"\(\s+", "(", plan_name)
            plan_name = re.sub(r"\s+\)", ")", plan_name)
            if plan_name:
                return plan_name
    return ""


def _extract_plan_name_from_lines(lines: list[str]) -> str:
    """Extract a plan name without consuming neighboring form fields."""
    excluded_phrases = (
        "name of plan",
        "plan name",
        "plan identification",
        "plan number",
        "plan year",
        "employee benefit plan",
        "annual return",
        "schedule h",
        "form 5500",
    )
    field_boundary = re.compile(
        r"^(?:EIN|employer\s+identification|plan\s+identification\s+number|"
        r"plan\s+number|plan\s+year)\b|^\d{2}-\d{7}\b|^\d{3}$",
        flags=re.IGNORECASE,
    )

    def clean_name(parts: list[str]) -> str:
        value = re.sub(r"\s+", " ", " ".join(parts)).strip(" :#-\t")
        value = re.sub(r"^.*[.!?]\s+\d+\s+(?=[A-Z])", "", value)
        uppercase_company = re.search(
            r"\b([A-Z][A-Z0-9&.,'() -]*\b(?:INC|LLC|CORP|CORPORATION|COMPANY|CO)\.?.*)",
            value,
        )
        if uppercase_company and any(character.islower() for character in value[:uppercase_company.start()]):
            value = uppercase_company.group(1)
        value = re.sub(r"\(\s+", "(", value)
        return re.sub(r"\s+\)", ")", value)

    def is_name_start(line: str) -> bool:
        if re.search(
            r"\b(?:inc|llc|corp|corporation|company|co)\.?\b|\b401\s*\(",
            line,
            flags=re.IGNORECASE,
        ):
            return True
        letters = [character for character in line if character.isalpha()]
        words = re.findall(r"[A-Za-z]+", line)
        return bool(
            len(words) >= 2
            and letters
            and sum(character.isupper() for character in letters) / len(letters) >= 0.8
        )

    def is_name_continuation(line: str) -> bool:
        return bool(
            not re.search(r"\d", line)
            and re.match(r"^(?:for|of|and|the)\b", line.strip(), flags=re.IGNORECASE)
            and len(re.findall(r"[A-Za-z]+", line)) >= 3
        )

    label_pattern = re.compile(r"(?:name\s+of\s+plan|plan\s+name)\b", re.IGNORECASE)
    for index, line in enumerate(lines):
        label_match = label_pattern.search(line)
        if label_match is None:
            continue
        parts = [line[label_match.end():].strip(" :#-\t")]
        following_lines = lines[index + 1:index + 4]
        for offset, following_line in enumerate(following_lines):
            if field_boundary.search(following_line.strip()):
                break
            parts.append(following_line.strip())
            candidate = clean_name([part for part in parts if part])
            if re.search(r"\bplan\b\s*$", candidate, flags=re.IGNORECASE):
                if offset + 1 < len(following_lines) and is_name_continuation(
                    following_lines[offset + 1]
                ):
                    parts.append(following_lines[offset + 1].strip())
                    return clean_name([part for part in parts if part])
                return candidate

    for index, line in enumerate(lines):
        if field_boundary.search(line.strip()) or not is_name_start(line):
            continue
        combined_parts = [line.strip()]
        following_lines = lines[index + 1:index + 4]
        for offset, following_line in enumerate(following_lines):
            if field_boundary.search(following_line.strip()):
                break
            combined_parts.append(following_line.strip())
            candidate = clean_name(combined_parts)
            has_organization_marker = bool(
                re.search(
                    r"\b(?:inc|llc|corp|corporation|company|co)\.?\b|\b401\s*\(",
                    candidate,
                    flags=re.IGNORECASE,
                )
            )
            if has_organization_marker and re.search(
                r"\bplan\b\s*$",
                candidate,
                flags=re.IGNORECASE,
            ):
                if offset + 1 < len(following_lines) and is_name_continuation(
                    following_lines[offset + 1]
                ):
                    combined_parts.append(following_lines[offset + 1].strip())
                    return clean_name(combined_parts)
                return candidate

    for line in lines:
        lower_line = line.lower()
        if any(lower_line.strip().startswith(phrase) for phrase in excluded_phrases):
            continue
        match = re.search(
            r"([A-Za-z][A-Za-z0-9&.,'() /-]*?\bplan\b)"
            r"(?=\s+(?:EIN|employer\s+identification|plan\s+identification\s+number|"
            r"plan\s+number|plan\s+year|\d{2}-\d{7}|\d{3}\b)|$)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return clean_name([match.group(1)])

    next_field = (
        r"(?=\s+(?:EIN|employer\s+identification|plan\s+identification\s+number|"
        r"plan\s+number|plan\s+year)\b|$)"
    )
    for line in lines:
        match = re.search(
            rf"(?:name\s+of\s+plan|plan\s+name)[ \t]*[:#-]?[ \t]*(.+?){next_field}",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip(" :#-\t")
            if value and not any(phrase in value.lower() for phrase in excluded_phrases):
                return value
    return ""


def extract_plan_metadata(
    doc: "fitz.Document",
    schedule_page_indices: list[int] | None = None,
) -> PlanMetadata:
    """Extract standard Form 5500 identifiers from filing and Schedule H pages."""
    candidate_indices = list(dict.fromkeys([0] + list(schedule_page_indices or [])))
    texts = []
    visual_lines = []
    for page_index in candidate_indices:
        if page_index < 0 or page_index >= len(doc):
            continue
        texts.append(_page_text_or_quick_ocr(doc, page_index))
        words, _, _ = extract_words(doc[page_index])
        visual_lines.extend(_build_visual_text_lines(words))

    raw_lines = [
        re.sub(r"\s+", " ", line).strip()
        for text in texts
        for line in text.splitlines()
        if line.strip()
    ]
    visual_lines = [re.sub(r"\s+", " ", line).strip() for line in visual_lines if line.strip()]
    lines = raw_lines + visual_lines
    combined_text = "\n".join(lines)

    plan_name = _extract_schedule_header_plan_name(raw_lines)
    if not plan_name:
        plan_name = _extract_plan_name_from_lines(lines)

    ein_label = r"(?:employer\s+identification\s+(?:number|no\.?)|\bEIN(?:\s+(?:number|no\.?))?)"
    ein_match = re.search(
        rf"{ein_label}[^\d\n]{{0,80}}(\d{{2}}\s*-?\s*\d{{7}})",
        combined_text,
        flags=re.IGNORECASE,
    )
    if ein_match is None:
        for index, line in enumerate(lines):
            if not re.search(ein_label, line, flags=re.IGNORECASE):
                continue
            nearby_text = " ".join(lines[index:index + 3])
            ein_match = re.search(r"\b(\d{2}\s*-?\s*\d{7})\b", nearby_text)
            if ein_match:
                break
    ein = ""
    if ein_match:
        ein_digits = re.sub(r"\D", "", ein_match.group(1))
        ein = f"{ein_digits[:2]}-{ein_digits[2:]}"

    plan_number_label = (
        r"(?:three-digit\s+plan\s+number|plan\s+number|plan\s+identification\s+number|"
        r"plan\s+(?:identification\s+)?no\.?|\bPN\b)"
    )
    plan_number_match = re.search(
        rf"{plan_number_label}"
        r"(?:\s*\(PN\))?[^\d\n]{0,40}(\d{3})\b",
        combined_text,
        flags=re.IGNORECASE,
    )
    if plan_number_match is None:
        for index, line in enumerate(lines):
            if not re.search(plan_number_label, line, flags=re.IGNORECASE):
                continue
            nearby_text = " ".join(lines[index:index + 3])
            plan_number_match = re.search(r"\b(\d{3})\b", nearby_text)
            if plan_number_match:
                break
    plan_number = plan_number_match.group(1) if plan_number_match else ""

    month = (
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    )
    day_first_date = rf"\d{{1,2}}[ ./-]+{month}[ ,./-]+\d{{2,4}}"
    month_first_date = rf"{month}[ .-]+\d{{1,2}}(?:st|nd|rd|th)?[,]?[ ]+\d{{2,4}}"
    numeric_date = r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    date_value = rf"(?:{day_first_date}|{month_first_date}|{numeric_date})"
    range_separator = r"(?:to|through|and\s+ending|ending|[-\u2013\u2014])"
    flattened_text = re.sub(r"\s+", " ", combined_text)
    date_range_match = re.search(
        rf"(?:plan\s+year[^\d]{{0,100}}?)?({date_value})\s*"
        rf"{range_separator}\s*({date_value})",
        flattened_text,
        flags=re.IGNORECASE,
    )
    if date_range_match:
        start_date = re.sub(r"\s+", " ", date_range_match.group(1)).strip()
        end_date = re.sub(r"\s+", " ", date_range_match.group(2)).strip()
        plan_year = f"{start_date} to {end_date}"
    else:
        plan_date_match = re.search(
            rf"(?:plan\s+year|(?:for\s+the\s+)?year\s+ended|as\s+of)"
            rf"[^\n]{{0,80}}?({date_value})",
            combined_text,
            flags=re.IGNORECASE,
        )
        if plan_date_match:
            plan_year = re.sub(r"\s+", " ", plan_date_match.group(1)).strip()
        else:
            plan_year_match = re.search(
                r"plan\s+year[^\n]{0,100}?\b((?:19|20)\d{2})\b",
                combined_text,
                flags=re.IGNORECASE,
            )
            plan_year = plan_year_match.group(1) if plan_year_match else ""

    return PlanMetadata(
        plan_name=plan_name,
        ein=ein,
        plan_number=plan_number,
        plan_year=plan_year,
    )


def _normalize_detection_text(text: str) -> str:
    """Flatten OCR quirks that stop heading regexes from matching."""
    if not text:
        return ""
    # Tesseract emits ligatures (e.g. "ﬁ") that break plain-ASCII patterns.
    cleaned = unicodedata.normalize("NFKD", text)
    return re.sub(r"\s+", " ", cleaned)


def _looks_like_schedule_h(text: str, page_index: int) -> bool:
    """Return True when a page's text carries a Schedule H / SOA heading."""
    text = _normalize_detection_text(text)
    if not text:
        return False

    if SCHEDULE_H_HEADING_RE.search(text):
        return True

    # Safe fallback for filings whose heading only says "Schedule of Assets".
    top_text = " ".join(text.splitlines()[:40])
    has_short_heading = bool(SCHEDULE_OF_ASSETS_FALLBACK_RE.search(top_text))
    has_identity_header = bool(IDENTITY_HEADER_RE.search(top_text)) or ("issuer" in top_text.lower())
    has_description_header = bool(DESCRIPTION_HEADER_RE.search(top_text))
    has_value_header = bool(VALUE_HEADER_RE.search(top_text))
    if has_short_heading and has_identity_header and has_description_header and has_value_header:
        logger.info(
            "Schedule H fallback detection accepted page %d via 'Schedule of Assets' heading + column headers",
            page_index + 1,
        )
        return True

    return False


def find_schedule_h_pages(doc):
    """Return page indices that look like Schedule H or Schedule of Assets pages."""
    pages = []

    for i in range(len(doc)):
        page = doc[i]

        # Rotate page BEFORE trying to detect Schedule H
        normalize_orientation(page)

        text = page.get_text("text") or ""
        scanned = len(text.strip()) < MIN_SEARCHABLE_TEXT_CHARS

        if scanned:
            words = ocr_words(page)
            text = " ".join([w[4] for w in words])

        if _looks_like_schedule_h(text, i):
            pages.append(i)
            continue

        # A single-orientation OCR pass can read the heading sideways on
        # landscape or tilted scans, so retry with the rotation-sweeping probe
        # before giving up on the page.
        if scanned:
            retry_text = _page_text_or_quick_ocr(doc, i)
            if retry_text and _looks_like_schedule_h(retry_text, i):
                logger.info(
                    "Schedule H detected on page %d only after rotation retry.",
                    i + 1,
                )
                pages.append(i)

    return pages


def _build_top_region_text(words, top_ratio: float = 0.40, page_height: float | None = None) -> str:
    """Build reading-order text from the top portion of a page."""
    if not words:
        return ""

    ratio = max(0.0, min(top_ratio, 1.0))
    max_y = max(float(w[3]) for w in words)

    use_page_height = bool(page_height and page_height > 0 and max_y <= (page_height * 1.5))
    top_limit = (page_height * ratio) if use_page_height else (max_y * ratio)

    top_words = [w for w in words if ((float(w[1]) + float(w[3])) / 2.0) <= top_limit]
    if not top_words:
        return ""

    top_words.sort(key=lambda w: (round(float(w[1]), 1), float(w[0])))
    return " ".join(str(w[4]) for w in top_words if str(w[4]).strip())


def is_normal_soa_page(page: "fitz.Page", top_ratio: float = 0.40) -> bool:
    """Return True when normal SOA headers are all present in the page's top region."""
    try:
        normalize_orientation(page)
        text = page.get_text("text") or ""
        words = page.get_text("words") if len(text.strip()) >= MIN_SEARCHABLE_TEXT_CHARS else ocr_words(page)
        top_text = _build_top_region_text(words, top_ratio=top_ratio, page_height=float(page.rect.height))

        has_schedule_heading = bool(SCHEDULE_H_HEADING_RE.search(top_text)) or bool(
            SCHEDULE_OF_ASSETS_FALLBACK_RE.search(top_text)
        )
        has_identity_header = bool(IDENTITY_HEADER_RE.search(top_text)) or ("issuer" in top_text.lower())
        has_description_header = bool(DESCRIPTION_HEADER_RE.search(top_text))
        has_value_header = bool(VALUE_HEADER_RE.search(top_text))

        return has_schedule_heading and has_identity_header and has_description_header and has_value_header
    except Exception as exc:
        logger.debug("Normal SOA top-region check failed on page %d: %s", page.number + 1, exc)
        return False


def _rotation_readability_score(page: "fitz.Page", rotation: int) -> int:
    """Score how "readable" a page looks at a candidate rotation.

    Falls back to a quick OCR pass when the page has no native text (i.e.
    it's a scanned page), so orientation detection works for scanned
    landscape pages just as well as for searchable ones.
    """
    original_rotation = page.rotation
    page.set_rotation(rotation)
    text = page.get_text("text") or ""
    if len(text.strip()) < MIN_SEARCHABLE_TEXT_CHARS:
        try:
            image = render_page_to_image(page, zoom=QUICK_OCR_ZOOM)
            text = pytesseract.image_to_string(image, config=OCR_TESSERACT_CONFIG)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Quick OCR during orientation scoring failed: %s", exc)
    page.set_rotation(original_rotation)

    score = 0
    if SCHEDULE_H_HEADING_RE.search(text):
        score += 500
    if IDENTITY_HEADER_RE.search(text) and DESCRIPTION_HEADER_RE.search(text):
        score += 300
    lines = [ln for ln in text.splitlines() if ln.strip()]
    score += min(len(lines), 50)
    return score


def auto_rotate_image_for_ocr(image):
    """Rotate a rendered page image upright using Tesseract OSD when possible."""
    try:
        osd = pytesseract.image_to_osd(image)
        logger.info("OSD OUTPUT:\n%s", osd)

        rotate = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                rotate = int(line.split(":")[1].strip())
                break

        if rotate:
            logger.info("OSD detected rotation=%d. Rotating image.", rotate)
            image = image.rotate(-rotate, expand=True)
    except Exception as exc:
        logger.debug(
            "OSD orientation detection failed: %s",
            exc
        )

    return image


def normalize_orientation(page: "fitz.Page") -> None:
    """Rotate landscape pages upright before text extraction begins."""
    try:
        rect = page.rect

        page_width = rect.width
        page_height = rect.height

        if page_width > page_height:
            logger.info(
                "Page %d is landscape. Rotating 90 degrees.",
                page.number + 1,
            )
            page.set_rotation(90)

    except Exception as exc:
        logger.warning(
            "Could not determine page orientation on page %d: %s",
            page.number + 1,
            exc,
        )


def is_searchable(page) -> bool:
    """Return True when a page already has enough native text to skip OCR."""
    text = page.get_text("text") or ""
    return len(text.strip()) >= MIN_SEARCHABLE_TEXT_CHARS


def _native_text_rotation_hint(page) -> int | None:
    """Return the OCR correction angle when most native text runs vertically."""
    try:
        vertical_directions = []
        line_count = 0
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                if not any(span.get("text", "").strip() for span in line.get("spans", [])):
                    continue
                line_count += 1
                dx, dy = line.get("dir", (1.0, 0.0))
                if abs(dy) > abs(dx):
                    vertical_directions.append(dy)

        if line_count < 5 or len(vertical_directions) / line_count < 0.75:
            return None

        return 90 if sum(vertical_directions) < 0 else 270
    except Exception as exc:
        logger.debug("Native text orientation check failed: %s", exc)
        return None


def _rotate_native_word_boxes(words, page, rotation_hint: int):
    """Rotate native word coordinates into the page's readable orientation."""
    width = float(page.rect.width)
    height = float(page.rect.height)
    if rotation_hint == 90:
        return [
            (height - y1, x0, height - y0, x1, *word)
            for x0, y0, x1, y1, *word in words
        ]
    if rotation_hint == 270:
        return [
            (y0, width - x1, y1, width - x0, *word)
            for x0, y0, x1, y1, *word in words
        ]
    return words


def extract_words(page):
    """Return word boxes, source, and whether OCR detected page skew."""
    if is_searchable(page):
        words = page.get_text("words")
        rotation_hint = _native_text_rotation_hint(page)
        if rotation_hint is not None:
            words = _rotate_native_word_boxes(words, page, rotation_hint)
        return words, "TEXT", False
    words, skewed = _ocr_words_with_skew(page)
    return words, "OCR", skewed


def _retry_numeric_text_fragments(image, words):
    repaired = list(words)
    candidates = []
    for index, word in enumerate(words):
        if index == 0 or not re.fullmatch(r"\d{3,}", word[4]):
            continue
        previous = words[index - 1]
        height = max(word[3] - word[1], previous[3] - previous[1])
        if (
            re.fullmatch(r"[A-Za-z]", previous[4])
            and abs(word[1] - previous[1]) < height / 2
            and 0 <= word[0] - previous[2] < height / 5
        ):
            candidates.append(index)

    for index in reversed(candidates):
        start = index - 1
        end = index + 1
        height = max(words[index][3] - words[index][1], 1)
        while end < len(words):
            following = words[end]
            if (
                not re.fullmatch(r"[A-Za-z]+", following[4])
                or abs(following[1] - words[index][1]) >= height / 2
                or following[0] - words[end - 1][2] > height
            ):
                break
            end += 1
        original = words[start:end]
        padding = max(4, int(height / 3))
        left = max(0, min(word[0] for word in original) - padding)
        top = max(0, min(word[1] for word in original) - padding)
        right = min(image.width, max(word[2] for word in original) + padding)
        bottom = min(image.height, max(word[3] for word in original) + padding)
        retry = pytesseract.image_to_data(
            image.crop((left, top, right, bottom)),
            config="--oem 3 --psm 7",
            output_type=pytesseract.Output.DICT,
        )
        retry_words = []
        for retry_index, text in enumerate(retry["text"]):
            text = text.strip()
            if not text:
                continue
            x0 = left + retry["left"][retry_index]
            y0 = top + retry["top"][retry_index]
            x1 = x0 + retry["width"][retry_index]
            y1 = y0 + retry["height"][retry_index]
            if retry_words and re.fullmatch(r"[A-Za-z]", retry_words[-1][4]):
                previous = retry_words[-1]
                if re.fullmatch(r"[A-Za-z]+", text) and x0 - previous[2] < height / 5:
                    retry_words[-1] = (
                        previous[0], min(previous[1], y0), x1,
                        max(previous[3], y1), previous[4] + text,
                    )
                    continue
            retry_words.append((x0, y0, x1, y1, text))
        retry_text = "".join(word[4] for word in retry_words).lower()
        if (
            retry_words
            and all(re.fullmatch(r"[A-Za-z]+", word[4]) for word in retry_words)
            and all(word[4].lower() in retry_text for word in original if word[4].isalpha())
        ):
            repaired[start:end] = retry_words
    return repaired


def _ocr_words_with_skew(page):
    """OCR a page and return word boxes plus its measured skew status."""
    zoom = OCR_ZOOM
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, -1)

    image = Image.fromarray(img)

    image = auto_rotate_image_for_ocr(image)

    image, skew_angle = deskew_image_with_angle(image)

    image = preprocess_for_ocr(image)

    processed = np.array(image)
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed

    data = pytesseract.image_to_data(
        gray,
        config=OCR_TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        words.append((x, y, x + w, y + h, text))

    return (
        _retry_numeric_text_fragments(image, words),
        abs(skew_angle) >= DESKEW_MIN_ANGLE_DEG,
    )


def ocr_words(page):
    """OCR a page and return word boxes in the same shape as PyMuPDF output."""
    return _ocr_words_with_skew(page)[0]


def _is_tickered_page(text: str) -> bool:
    """Detect whether a page's text looks like a "tickered" format.

    Heuristic: look at the first ~40 lines and ensure a set of required
    tokens appear there. Returns False for empty or very short text.
    """
    if not text:
        return False

    top_text = " ".join(text.splitlines()[:40]).lower()

    # Guardrail: do not classify classic Schedule H grid/non-tickered headers
    # as tickered pages. Rotated OCR can include words like "maturity date"
    # and "current value" on normal grid tables.
    non_tickered_markers = [
        "identity of issuer",
        "description of investment including",
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "schedule h, line 4i",
    ]
    if any(marker in top_text for marker in non_tickered_markers):
        return False

    required = [
        "investment option",
        "maturity date",
        "interest rate",
        #  "cost of assets",
        "current value",
    ]
    present = [token for token in required if token in top_text]

    # Accept only when the explicit tickered anchor exists.
    # This avoids false positives on standard/grid Schedule H pages.
    if "investment option" in top_text and "current value" in top_text:
        logger.debug("Tickered check: detected required pair tokens: %s", present)
        return True

    missing = [token for token in required if token not in top_text]
    logger.debug("Tickered check: insufficient tokens present=%s missing=%s", present, missing)
    return False


def is_tickered_document(doc: "fitz.Document", pages_to_check: int = 8) -> bool:
    """Scan the first `pages_to_check` pages for the tickered pattern.

    Uses the cheap page-text / quick OCR probe to remain fast across many
    documents.
    """
    max_pages = min(len(doc), pages_to_check)
    for i in range(max_pages):
        text = _page_text_or_quick_ocr(doc, i)
        if _is_tickered_page(text):
            return True
    return False


def is_tickered_among_pages(doc: "fitz.Document", page_indices: list[int]) -> bool:
    """Check whether any of the provided page indices appears to be tickered.

    This inspects each page using the same cheap OCR probe so rotated
    scanned pages are still detectable.
    """
    for i in page_indices:
        if i < 0 or i >= len(doc):
            continue
        text = _page_text_or_quick_ocr(doc, i)
        if _is_tickered_page(text):
            return True
    return False


def get_tickered_page_indices(doc: "fitz.Document", page_indices: list[int]) -> list[int]:
    """Return the subset of page indices that appear to use tickered layout."""
    tickered_pages: list[int] = []
    for i in page_indices:
        if i < 0 or i >= len(doc):
            continue
        text = _page_text_or_quick_ocr(doc, i)
        if _is_tickered_page(text):
            tickered_pages.append(i)
    return tickered_pages
