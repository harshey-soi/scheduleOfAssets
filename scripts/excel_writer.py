"""Writes extracted Schedule H data to an Excel workbook, one worksheet per
source page, named after that page's page number.
"""
from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from typing import List

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.worksheet import Worksheet

from models import PlanMetadata, SchedulePageResult

logger = logging.getLogger(__name__)

_HEADER_FILL = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
_HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
_LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
_RIGHT = Alignment(horizontal="right", vertical="center", wrap_text=True)
_NOTICE_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
_NOTICE_FONT = Font(bold=True, color="7F6000", size=10)
_NOTICE_ALIGNMENT = Alignment(horizontal="left", vertical="center", wrap_text=True)
_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
_GRID_DIVIDER_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thick", color="000000"),
)

_THREE_COL_HEADERS = ["Identity", "Description", "Current Value"]
_FOUR_COL_HEADERS = ["Identity", "Description", "Cost", "Current Value"]
_THREE_COL_WIDTHS = [32, 55, 20]
_FOUR_COL_WIDTHS = [30, 45, 16, 20]
_SKEW_NOTICE = (
    "Special-Case File Disclaimer: Due to document quality and page alignment challenges, "
    "certain extracted data may require manual validation. While every effort has been "
    "made to ensure accurate extraction, minor discrepancies in text recognition, field "
    "placement, or Participant Loan information may occur. We recommend reviewing the "
    "extracted output before final use."
)
_LEGEND_TICKER_RE = re.compile(r"\b[A-Z0-9]{5,10}\b")


def _to_excel_number(value):
    """Convert monetary text to an integer or decimal Excel value."""
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str) or not value.strip():
        return value

    text = value.strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[$,\s]", "", text)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return value
    numeric_value = int(number) if number == number.to_integral_value() else float(number)
    return -numeric_value if negative else numeric_value


def _parse_ticker_legend(lines: List[str]):
    """Split legend text into ticker mappings, definitions, and fallback text."""
    entries: List[List[str]] = []
    definitions: List[str] = []
    fallback_lines: List[str] = []

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        lower_line = line.lower()
        if not line or lower_line.rstrip(":") in {"legend", "key", "investment option"}:
            continue
        if lower_line.startswith(("cost of assets", "current value")):
            definitions.append(line)
            continue

        ticker_matches = [
            match
            for match in _LEGEND_TICKER_RE.finditer(line)
            if any(character.isdigit() for character in match.group(0))
            or match.group(0).endswith("X")
        ]
        if ticker_matches:
            for index, match in enumerate(ticker_matches):
                description_end = (
                    ticker_matches[index + 1].start()
                    if index + 1 < len(ticker_matches)
                    else len(line)
                )
                description = line[match.end():description_end].strip(" :-\t")
                entries.append([match.group(0), description])
            continue

        if entries and not definitions:
            entries[-1][1] = f"{entries[-1][1]} {line}".strip()
        elif definitions:
            definitions[-1] = f"{definitions[-1]} {line}".strip()
        else:
            fallback_lines.append(line)

    return entries, definitions, fallback_lines


def _write_page_sheet(
    worksheet: Worksheet,
    page: SchedulePageResult,
    metadata: PlanMetadata,
    divider_after_rows: set[int] | None = None,
) -> None:
    """Render one extracted page into a worksheet with consistent styling."""
    # Use 'Investment' as the identity header for tickered layouts
    identity_label = "Investment" if (page.source or "").upper().startswith("TICKERED") else "Identity"
    if page.has_cost_column:
        headers = [_FOUR_COL_HEADERS[0].replace("Identity", identity_label)] + _FOUR_COL_HEADERS[1:]
    else:
        headers = [_THREE_COL_HEADERS[0].replace("Identity", identity_label)] + _THREE_COL_HEADERS[1:]
    widths = _FOUR_COL_WIDTHS if page.has_cost_column else _THREE_COL_WIDTHS
    numeric_column_indices = {len(headers) - 1, len(headers)} if page.has_cost_column else {len(headers)}
    metadata_rows = (
        ("Plan Name", metadata.plan_name),
        ("EIN", metadata.ein),
        ("Plan Number", metadata.plan_number),
        ("Plan Year", metadata.plan_year),
    )
    for row_index, (label, value) in enumerate(metadata_rows, start=1):
        label_cell = worksheet.cell(row=row_index, column=1, value=label)
        label_cell.fill = _HEADER_FILL
        label_cell.font = _HEADER_FONT
        label_cell.alignment = _LEFT
        label_cell.border = _BORDER

        value_cell = worksheet.cell(row=row_index, column=2, value=value)
        value_cell.alignment = _LEFT
        value_cell.border = _BORDER

    header_row = 6

    if page.is_skewed:
        worksheet.merge_cells(
            start_row=header_row,
            start_column=1,
            end_row=header_row,
            end_column=len(headers),
        )
        notice_cell = worksheet.cell(row=header_row, column=1, value=_SKEW_NOTICE)
        notice_cell.fill = _NOTICE_FILL
        notice_cell.font = _NOTICE_FONT
        notice_cell.alignment = _NOTICE_ALIGNMENT
        notice_cell.border = _BORDER
        worksheet.row_dimensions[header_row].height = 72
        header_row += 1

    for col_idx, header in enumerate(headers, start=1):
        cell = worksheet.cell(row=header_row, column=col_idx, value=header)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = _CENTER
        cell.border = _BORDER

    dividers = divider_after_rows or set()
    for data_row_index, row in enumerate(page.rows, start=1):
        row_idx = header_row + data_row_index
        for col_idx, value in enumerate(row.to_list(include_cost=page.has_cost_column), start=1):
            if col_idx in numeric_column_indices:
                value = _to_excel_number(value)
            cell = worksheet.cell(row=row_idx, column=col_idx, value=value)
            cell.border = _GRID_DIVIDER_BORDER if data_row_index in dividers else _BORDER
            cell.alignment = _RIGHT if col_idx in numeric_column_indices else _LEFT
            if col_idx in numeric_column_indices and isinstance(value, int):
                cell.number_format = "#,##0;[Red](#,##0)"
            elif col_idx in numeric_column_indices and isinstance(value, float):
                cell.number_format = "#,##0.00;[Red](#,##0.00)"

    if page.legend:
        legend_row = header_row + len(page.rows) + 2
        worksheet.merge_cells(
            start_row=legend_row,
            start_column=1,
            end_row=legend_row,
            end_column=4,
        )
        legend_header = worksheet.cell(row=legend_row, column=1, value="Legend")
        legend_header.fill = _HEADER_FILL
        legend_header.font = _HEADER_FONT
        legend_header.alignment = _LEFT
        legend_header.border = _BORDER

        entries, definitions, fallback_lines = _parse_ticker_legend(page.legend)
        current_row = legend_row + 1

        if entries:
            worksheet.merge_cells(
                start_row=current_row,
                start_column=1,
                end_row=current_row,
                end_column=4,
            )
            option_header = worksheet.cell(row=current_row, column=1, value="Investment Option")
            option_header.font = Font(bold=True)
            option_header.alignment = _LEFT
            current_row += 1

            midpoint = (len(entries) + 1) // 2
            left_entries = entries[:midpoint]
            right_entries = entries[midpoint:]
            for index, left_entry in enumerate(left_entries):
                row_entries = [(1, left_entry)]
                if index < len(right_entries):
                    row_entries.append((3, right_entries[index]))
                for column, (ticker, description) in row_entries:
                    ticker_cell = worksheet.cell(row=current_row, column=column, value=ticker)
                    ticker_cell.font = Font(bold=True)
                    ticker_cell.alignment = _LEFT
                    description_cell = worksheet.cell(
                        row=current_row,
                        column=column + 1,
                        value=description,
                    )
                    description_cell.alignment = _LEFT
                current_row += 1

        for text in fallback_lines:
            worksheet.merge_cells(
                start_row=current_row,
                start_column=1,
                end_row=current_row,
                end_column=4,
            )
            legend_cell = worksheet.cell(row=current_row, column=1, value=text)
            legend_cell.alignment = _LEFT
            worksheet.row_dimensions[current_row].height = max(18, 15 * (len(text) // 100 + 1))
            current_row += 1

        for text in definitions:
            label, separator, explanation = text.partition(":")
            label_cell = worksheet.cell(
                row=current_row,
                column=1,
                value=f"{label.strip()}:" if separator else label.strip(),
            )
            label_cell.font = Font(bold=True)
            label_cell.alignment = _LEFT
            worksheet.merge_cells(
                start_row=current_row,
                start_column=2,
                end_row=current_row,
                end_column=4,
            )
            explanation_cell = worksheet.cell(
                row=current_row,
                column=2,
                value=explanation.strip() if separator else "",
            )
            explanation_cell.alignment = _LEFT
            worksheet.row_dimensions[current_row].height = max(18, 15 * (len(text) // 100 + 1))
            current_row += 1

    for col_idx, width in enumerate(widths, start=1):
        worksheet.column_dimensions[chr(ord("A") + col_idx - 1)].width = width
    if page.legend:
        worksheet.column_dimensions["D"].width = max(45, worksheet.column_dimensions["D"].width)


def _unique_sheet_name(desired: str, used_names: set) -> str:
    """Excel worksheet names must be unique (and <= 31 chars). Page numbers
    already satisfy the length limit, but guard uniqueness defensively.
    """
    name = desired[:31]
    suffix = 1
    while name in used_names:
        suffix += 1
        candidate = f"{desired}_{suffix}"
        name = candidate[:31]
    return name


def _sheet_name_for_page(page: SchedulePageResult) -> str:
    """Build a worksheet name containing the layout and PDF page number."""
    source = (page.source or "").upper()
    if source.startswith("TICKERED"):
        layout = "TICKERED"
    elif source == "GRID":
        layout = "GRID"
    else:
        layout = "NON_GRID"
    return f"{layout}_{page.page_number}"


def _group_consecutive_grid_pages(
    pages: List[SchedulePageResult],
) -> List[List[SchedulePageResult]]:
    """Group adjacent grid pages while leaving every other page independent."""
    groups: List[List[SchedulePageResult]] = []
    for page in pages:
        is_grid = (page.source or "").upper() == "GRID"
        if (
            is_grid
            and groups
            and (groups[-1][-1].source or "").upper() == "GRID"
            and page.page_number == groups[-1][-1].page_number + 1
        ):
            groups[-1].append(page)
        else:
            groups.append([page])
    return groups


def write_workbook(
    pages: List[SchedulePageResult],
    output_path: str,
    metadata: PlanMetadata | None = None,
) -> None:
    """Write one worksheet per non-empty Schedule H page to `output_path`."""
    non_empty_pages = [page for page in pages if not page.is_empty()]
    if not non_empty_pages:
        raise ValueError("No non-empty Schedule H pages to write.")

    workbook = Workbook()
    workbook.remove(workbook.active)

    used_names: set = set()
    workbook_metadata = metadata or PlanMetadata()
    for page_group in _group_consecutive_grid_pages(non_empty_pages):
        first_page = page_group[0]
        divider_after_rows: set[int] = set()
        if len(page_group) > 1:
            combined_rows = []
            row_count = 0
            for page in page_group:
                combined_rows.extend(page.rows)
                row_count += len(page.rows)
                if page is not page_group[-1] and page.rows:
                    divider_after_rows.add(row_count)
            page = SchedulePageResult(
                page_number=first_page.page_number,
                rows=combined_rows,
                has_cost_column=any(item.has_cost_column for item in page_group),
                source="GRID",
                is_skewed=any(item.is_skewed for item in page_group),
            )
            desired_name = f"GRID_{first_page.page_number}-{page_group[-1].page_number}"
        else:
            page = first_page
            desired_name = _sheet_name_for_page(page)

        sheet_name = _unique_sheet_name(desired_name, used_names)
        used_names.add(sheet_name)
        worksheet = workbook.create_sheet(title=sheet_name)
        _write_page_sheet(
            worksheet,
            page,
            workbook_metadata,
            divider_after_rows=divider_after_rows,
        )

    workbook.save(output_path)
    logger.info("Workbook written: %s", output_path)


def write_error_workbook(output_path: str, error_text: str) -> None:
    """Write a one-sheet workbook that captures a processing failure."""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Error"
    # Write the error text into A1 and enable wrap
    cell = worksheet.cell(row=1, column=1, value=error_text)
    cell.alignment = Alignment(wrap_text=True)
    # widen the column so the text is readable
    worksheet.column_dimensions["A"].width = 120
    workbook.save(output_path)
    logger.info("Error workbook written: %s", output_path)
