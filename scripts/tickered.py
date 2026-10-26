"""Tickered-format Schedule of Assets extraction helpers.

This adapts the tickered extraction you provided to the project's
PDF handling: it uses `pdf_processor.render_page_to_image` for raster
rendering and `utils.is_numeric_value` for numeric checks. The public
entry point `process_tickered_pdf(pdf_path: str)` returns an
`ExtractionResult` compatible with the normal pipeline.
"""
from __future__ import annotations

import logging
import re
from typing import List

import pytesseract

from config import DESKEW_MIN_ANGLE_DEG
from pdf_processor import render_page_to_image, open_document
from ocr_preprocessing import deskew_image_with_angle
from utils import is_numeric_value
from models import ExtractionResult, SchedulePageResult, ScheduleRow

logger = logging.getLogger(__name__)


def _is_footer_line(lower_line):
    """Return True when a line clearly belongs to footer or report boilerplate."""
    footer_keywords = [
        "accompanying report",
        "notes to the financial statements",
        "see independent auditor's report",
        "attachment",
        "attached",
        "supplemental schedule",
        "the accompanying notes",
        "form 5500",
        "ein #",
        "ein:",
        "employer identification number",
    ]
    return any(keyword in lower_line for keyword in footer_keywords)


def _is_legend_start(lower_line):
    """Return True when a line begins the explanatory legend section."""
    stripped = lower_line.strip()
    if "legend" in stripped or stripped.startswith(("key:", "notes:")):
        return True
    return stripped.startswith("*") and any(
        marker in stripped
        for marker in ("indicates", "party-in-interest", "participant directed")
    )


def _is_non_table_line(lower_line):
    """Return True for lines that mark the end of tickered table content."""
    if _is_footer_line(lower_line) or _is_legend_start(lower_line):
        return True
    if "attachment" in lower_line or "attached" in lower_line:
        return True
    if re.search(r"\bein\s*#?\s*\d", lower_line):
        return True
    if "tax id" in lower_line or "federal id" in lower_line:
        return True
    return False


def _is_table_amount_token(token):
    """Treat numeric tokens as values while rejecting likely year-only fields."""
    token = str(token or "").strip()
    if not token:
        return False
    if not is_numeric_value(token):
        return False

    compact = token.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()
    if not compact.isdigit():
        return True

    if len(compact) <= 4 and compact.isdigit() and 1900 <= int(compact) <= 2100:
        return False
    return True


def _split_footer(lines):
    """Trim trailing footer text once footer boilerplate begins."""
    kept = []
    for line in lines:
        lower = line.lower()
        if _is_footer_line(lower):
            break
        kept.append(line)
    return kept


def _extract_legend_lines(lines):
    """Return legend text without including trailing report boilerplate."""
    start_idx = next(
        (index for index, line in enumerate(lines) if _is_legend_start(line.lower())),
        None,
    )
    if start_idx is None:
        return []

    legend_lines = []
    for line in lines[start_idx:]:
        if legend_lines and _is_footer_line(line.lower()):
            break
        if re.fullmatch(r"(?:page\s+)?\d+", line.lower()):
            continue
        if not legend_lines or line != legend_lines[-1]:
            legend_lines.append(line)
    return legend_lines


def _extract_trailing_amount_tokens(text, max_count=2):
    """Split a line into its leading text and up to two trailing amount tokens."""
    matches = list(re.finditer(r"\(?\$?\d[\d,]*(?:\.\d+)?\)?", text))
    if not matches:
        return text.strip(), []

    amounts = []
    left_end = len(text)
    for match in reversed(matches):
        token = match.group(0).strip()
        if not is_numeric_value(token):
            continue
        amounts.append(token)
        left_end = match.start()
        if len(amounts) >= max_count:
            break

    amounts.reverse()
    return text[:left_end].strip(" -:\t"), amounts


def _extract_tickered_row_from_line(line, allow_blank_investment=False):
    """Parse one tickered line into investment, maturity, rate, cost, and value."""
    left_text, amounts = _extract_trailing_amount_tokens(line, max_count=2)
    if not amounts:
        return None

    if len(amounts) >= 2:
        cost, value = amounts[-2], amounts[-1]
    else:
        cost, value = "", amounts[-1]

    if not _is_table_amount_token(cost):
        cost = ""
    if not _is_table_amount_token(value):
        value = ""

    interest_match = re.search(r"\b\d+(?:\.\d+)?%\b", left_text)
    interest_rate = interest_match.group(0) if interest_match else ""

    date_match = re.search(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4}\b",
        left_text,
        flags=re.IGNORECASE,
    )
    maturity_date = date_match.group(0) if date_match else ""

    investment = left_text
    if maturity_date:
        investment = re.sub(re.escape(maturity_date), "", investment, count=1, flags=re.IGNORECASE)
    if interest_rate:
        investment = re.sub(re.escape(interest_rate), "", investment, count=1, flags=re.IGNORECASE)
    investment = re.sub(r"\s{2,}", " ", investment).strip(" -:\t")

    if not investment and not maturity_date and not interest_rate:
        if allow_blank_investment and (cost or value):
            return ["", "", "", cost, value]
        return None

    if not value and not cost:
        return [investment, maturity_date, interest_rate, "", ""]

    return [investment, maturity_date, interest_rate, cost, value]


def _extract_tickered_content_from_text(page_text):
    """Extract tickered table rows and separate legend text."""
    all_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    legend_lines = _extract_legend_lines(all_lines)
    lines = _split_footer(all_lines)

    start_idx = -1
    for idx, line in enumerate(lines):
        lower = line.lower()
        if "investment option" in lower and "current value" in lower:
            # Log header line and small context for debugging
            try:
                ctx_start = max(0, idx - 3)
                ctx_end = min(len(lines), idx + 4)
                header_context = [lines[i] for i in range(ctx_start, ctx_end)]
                logger.info(
                    "Tickered detected header at line %d (context %d..%d): %s",
                    idx,
                    ctx_start,
                    ctx_end - 1,
                    header_context,
                )
            except Exception:
                logger.debug("Failed to log tickered header context at idx=%d", idx)
            start_idx = idx + 1
            break
    if start_idx == -1:
        return [], legend_lines

    lines = lines[start_idx:]

    rows = []
    in_participant_loans_section = False
    for idx, line in enumerate(lines):
        lower = line.lower()
        if _is_non_table_line(lower):
            break
        if "participant" in lower and "loan" in lower:
            in_participant_loans_section = True
        if any(token in lower for token in ["maturity date", "interest rate", "cost of assets", "current value"]):
            continue
        if lower.startswith("schedule h"):
            continue
        if lower.startswith("page ") or re.fullmatch(r"\d+", lower):
            continue
        if "legend" in lower or lower.startswith("*"):
            continue

        # If the next line looks like a numeric-only continuation (common when
        # OCR breaks amounts onto the following visual line), merge it and try
        # parsing the combined text as well.
        parsed = _extract_tickered_row_from_line(
            line,
            allow_blank_investment=not in_participant_loans_section,
        )
        if not parsed:
            continue
        rows.append(parsed)

    deduped = []
    for row in rows:
        if deduped and row == deduped[-1]:
            continue
        deduped.append(row)
    return deduped, legend_lines


def _extract_tickered_rows_from_text(page_text):
    """Extract only tickered rows for callers that do not need legend text."""
    rows, _ = _extract_tickered_content_from_text(page_text)
    return rows


def extract_tickered_rows_from_page(doc, page_index):
    """Extract tickered rows from a `fitz.Document` page index.

    Tries native text first and falls back to OCR on rendered images
    (including rotated attempts) if needed.
    """
    page = doc[page_index]
    page_text = page.get_text("text") or ""

    searchable_rows, searchable_legend = _extract_tickered_content_from_text(page_text)
    if searchable_rows or searchable_legend:
        return searchable_rows, searchable_legend, "TICKERED_TEXT", False

    try:
        image = render_page_to_image(page, zoom=300 / 72)
        image, skew_angle = deskew_image_with_angle(image)
        attempts = [
            ("original", image),
            ("rotated_90", image.rotate(90, expand=True)),
            ("rotated_270", image.rotate(270, expand=True)),
        ]

        best_rows = []
        best_legend = []
        best_label = "original"
        for label, attempt_image in attempts:
            text = pytesseract.image_to_string(
                attempt_image,
                config=r'--oem 3 --psm 6 -c preserve_interword_spaces=0',
            )
            rows, legend = _extract_tickered_content_from_text(text)
            score = (len(rows), sum(len(line) for line in legend))
            best_score = (len(best_rows), sum(len(line) for line in best_legend))
            if score > best_score:
                best_rows = rows
                best_legend = legend
                best_label = label

        if best_rows and best_label != "original":
            logger.info("Used %s orientation for tickered extraction on page %d.", best_label, page_index + 1)
        return best_rows, best_legend, "TICKERED_OCR", abs(skew_angle) >= DESKEW_MIN_ANGLE_DEG
    except Exception as error:
        logger.warning("Tickered OCR failed on page %d: %s", page_index + 1, error)
        return [], [], "TICKERED_OCR", False


def process_tickered_pdf(
    pdf_path: str,
    page_indices: List[int] | None = None,
) -> ExtractionResult:
    """High-level entry point: open PDF, extract selected tickered rows per page,
    and return an `ExtractionResult` compatible with the rest of the
    pipeline.
    """
    doc = open_document(pdf_path)
    try:
        if page_indices is None:
            page_indices = list(range(len(doc)))

        plan_name = "Unknown Plan Name"
        # Try to extract plan name from first non-blank line of page 1
        if doc.page_count > 0:
            text = doc[0].get_text("text") or ""
            for line in text.splitlines():
                line = line.strip()
                if line:
                    plan_name = line
                    break

        page_results: List[SchedulePageResult] = []
        for i in page_indices:
            rows, legend, source, is_skewed = extract_tickered_rows_from_page(doc, i)
            schedule_rows = []
            for parsed in rows:
                investment, maturity_date, interest_rate, cost, value = parsed
                description_parts = []
                if maturity_date:
                    description_parts.append(f"Maturity: {maturity_date}")
                if interest_rate:
                    description_parts.append(f"Interest: {interest_rate}")
                description = " ".join(description_parts).strip()

                schedule_row = ScheduleRow(
                    identity=investment,
                    description=description,
                    cost=cost or None,
                    current_value=value or "",
                )
                schedule_rows.append(schedule_row)

            page_result = SchedulePageResult(
                page_number=i + 1,
                rows=schedule_rows,
                has_cost_column=any(r.cost for r in schedule_rows),
                source=source,
                is_skewed=is_skewed,
                legend=legend,
            )
            page_results.append(page_result)

        return ExtractionResult(plan_name=plan_name, pages=page_results)
    finally:
        doc.close()
