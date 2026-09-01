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
from typing import List, Tuple
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2

from config import (
    DESCRIPTION_HEADER_RE,
    IDENTITY_HEADER_RE,
    MIN_SEARCHABLE_TEXT_CHARS,
    OCR_TESSERACT_CONFIG,
    OCR_ZOOM,
    QUICK_OCR_ZOOM,
    SCHEDULE_OF_ASSETS_FALLBACK_RE,
    SCHEDULE_H_HEADING_RE,
)
from ocr_preprocessing import preprocess_for_ocr
from utils import sanitize_filename

logger = logging.getLogger(__name__)


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
    pages = []

    for i in range(len(doc)):
        page = doc[i]

        # Rotate page BEFORE trying to detect Schedule H
        normalize_orientation(page)

        text = page.get_text("text")

        if len(text.strip()) < MIN_SEARCHABLE_TEXT_CHARS:
            words = ocr_words(page)
            text = " ".join([w[4] for w in words])

        if SCHEDULE_H_HEADING_RE.search(text):
            pages.append(i)
            continue

        # Safe fallback for filings whose heading only says "Schedule of Assets".
        # Require the short heading plus recognizable Schedule H column headers,
        # so we do not over-match unrelated pages.
        top_text = " ".join(text.splitlines()[:40])
        has_short_heading = bool(SCHEDULE_OF_ASSETS_FALLBACK_RE.search(top_text))
        has_identity_header = bool(IDENTITY_HEADER_RE.search(top_text))
        has_description_header = bool(DESCRIPTION_HEADER_RE.search(top_text))
        has_value_header = ("current value" in top_text.lower()) or ("cost" in top_text.lower())
        if has_short_heading and has_identity_header and has_description_header and has_value_header:
            logger.info(
                "Schedule H fallback detection accepted page %d via 'Schedule of Assets' heading + column headers",
                i + 1,
            )
            pages.append(i)

    return pages


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
    """
    Use Tesseract OSD to detect page orientation and rotate
    image upright before OCR.
    """
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
    """
    Determine if page has enough native text to skip OCR.
    """
    text = page.get_text("text") or ""
    return len(text.strip()) >= MIN_SEARCHABLE_TEXT_CHARS

def extract_words(page):
    if is_searchable(page):
        return page.get_text("words"), "TEXT"
    else:
        return ocr_words(page), "OCR"

def ocr_words(page):
    zoom = OCR_ZOOM
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, -1)

    # ✅ APPLY PREPROCESSING (CRITICAL)
    # image = Image.fromarray(img)
    # image = preprocess_for_ocr(image)
    image = Image.fromarray(img)

    image = auto_rotate_image_for_ocr(image)

    image = preprocess_for_ocr(image)

    processed = np.array(image)
    # ✅ FIX: handle both grayscale and RGB safely
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
    required = [
        "investment option",
        "maturity date",
        "interest rate",
      #  "cost of assets",
        "current value",
    ]
    present = [token for token in required if token in top_text]

    # Accept when the clear pair of headers is present (most robust):
    if "investment option" in top_text and "current value" in top_text:
        logger.debug("Tickered check: detected required pair tokens: %s", present)
        return True

    # Otherwise accept if a majority of tokens appear (tolerant fallback).
    if len(present) >= 3:
        logger.debug("Tickered check: majority tokens present: %s", present)
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