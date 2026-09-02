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
from typing import List, Tuple
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import config as config_module

from config import (
    DESCRIPTION_HEADER_RE,
    IDENTITY_HEADER_RE,
    MIN_SEARCHABLE_TEXT_CHARS,
    OCR_TESSERACT_CONFIG,
    OCR_ZOOM,
    QUICK_OCR_ZOOM,
    SCHEDULE_H_HEADING_RE,
    VALUE_HEADER_RE,
)
from ocr_preprocessing import preprocess_for_ocr
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


def find_schedule_h_pages(doc):
    """Return page indices that look like Schedule H or Schedule of Assets pages."""
    pages = []

    for i in range(len(doc)):
        page = doc[i]

        # Rotate page BEFORE trying to detect Schedule H
        normalize_orientation(page)

        text = page.get_text("text") or ""

        if len(text.strip()) < MIN_SEARCHABLE_TEXT_CHARS:
            words = ocr_words(page)
            text = " ".join([w[4] for w in words])

        if SCHEDULE_H_HEADING_RE.search(text):
            pages.append(i)
            continue

        # Safe fallback for filings whose heading only says "Schedule of Assets".
        top_text = " ".join(text.splitlines()[:40])
        has_short_heading = bool(SCHEDULE_OF_ASSETS_FALLBACK_RE.search(top_text))
        has_identity_header = bool(IDENTITY_HEADER_RE.search(top_text)) or ("issuer" in top_text.lower())
        has_description_header = bool(DESCRIPTION_HEADER_RE.search(top_text))
        has_value_header = bool(VALUE_HEADER_RE.search(top_text))
        if has_short_heading and has_identity_header and has_description_header and has_value_header:
            logger.info(
                "Schedule H fallback detection accepted page %d via 'Schedule of Assets' heading + column headers",
                i + 1,
            )
            pages.append(i)

    return pages


def _build_top_region_text(words, top_ratio: float = 0.35, page_height: float | None = None) -> str:
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


def is_normal_soa_page(page: "fitz.Page", top_ratio: float = 0.35) -> bool:
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

        if rotate != 0:
            logger.info(
                "OSD detected rotation=%d. Rotating image.",
                rotate
            )

            image = image.rotate(
                -rotate,
                expand=True
            )

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

def extract_words(page):
    """Return word boxes from native text when possible, otherwise from OCR."""
    if is_searchable(page):
        return page.get_text("words"), "TEXT"
    else:
        return ocr_words(page), "OCR"

def ocr_words(page):
    """OCR a page and return word boxes in the same shape as PyMuPDF output."""
    zoom = OCR_ZOOM
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, -1)

    image = Image.fromarray(img)

    image = auto_rotate_image_for_ocr(image)

    image = preprocess_for_ocr(image)

    processed = np.array(image)
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed


    data = pytesseract.image_to_data(
        gray,
        config=OCR_TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT
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

    return words


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