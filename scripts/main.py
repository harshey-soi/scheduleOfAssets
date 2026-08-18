"""Entry point: convert every PDF in the input folder into a Schedule H
Excel workbook (named after the plan) in the output folder.

Usage:
    python main.py
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import List

from config import FAILED_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER
from excel_writer import write_workbook
from models import ExtractionResult, SchedulePageResult
from pdf_processor import (
    extract_plan_name,
    extract_words,
    find_schedule_h_pages,
    normalize_orientation,
    open_document,
    is_tickered_document,
    is_tickered_among_pages,
)
import tickered
import grid_extractor
from schedule_parser import parse_schedule_h_page
from utils import configure_logging, sanitize_filename

logger = logging.getLogger(__name__)


def process_pdf(pdf_path: str) -> ExtractionResult:
    """Run the full extraction pipeline for a single PDF."""
    doc = open_document(pdf_path)
    try:
        plan_name = extract_plan_name(doc)
        logger.info("Plan name: %s", plan_name)

        target_pages = find_schedule_h_pages(doc)
        if not target_pages:
            raise ValueError("No Schedule H pages were found in this PDF.")
        logger.info("Schedule H found on page(s): %s", [p + 1 for p in target_pages])

        page_results: List[SchedulePageResult] = []
        for page_index in target_pages:
            page = doc[page_index]
            normalize_orientation(page)
            words, source = extract_words(page)
            rows, has_cost_column = parse_schedule_h_page(words)

            page_result = SchedulePageResult(
                page_number=page_index + 1,
                rows=rows,
                has_cost_column=has_cost_column,
                source=source,
            )
            layout_desc = (
                "4-col: Identity/Description/Cost/Current Value"
                if has_cost_column
                else "3-col: Identity/Description/Current Value"
            )
            logger.info(
                "Page %d (%s): extracted %d row(s) [%s]",
                page_result.page_number,
                source,
                len(rows),
                layout_desc,
            )
            page_results.append(page_result)

        return ExtractionResult(plan_name=plan_name, pages=page_results)
    finally:
        doc.close()


def _unique_output_path(directory: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base} ({counter}){ext}")
        counter += 1
    return candidate


def process_input_file(pdf_path: str) -> bool:
    """Process one PDF end-to-end; returns True on success."""
    try:
        logger.info("Processing: %s", pdf_path)
        # Open the document briefly to decide which pipeline to use.
        doc = open_document(pdf_path)
        try:
            # First, locate Schedule H pages so we can inspect the specific
            # pages for tickered layout (rotated/tickered pages are often
            # missed by a simple document-wide probe).
            target_pages = find_schedule_h_pages(doc)
            if target_pages and is_tickered_among_pages(doc, target_pages):
                logger.info("Detected tickered layout on Schedule H pages: %s", pdf_path)
                result = tickered.process_tickered_pdf(pdf_path)
            elif target_pages and grid_extractor.is_grid_among_pages(doc, target_pages):
                logger.info("Detected grid table layout on Schedule H pages: %s", pdf_path)
                result = grid_extractor.process_grid_pdf(pdf_path, page_indices=target_pages)
            else:
                result = process_pdf(pdf_path)
        finally:
            doc.close()

        if result.is_empty():
            raise ValueError("No data extracted from any Schedule H page.")

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_filename = f"{sanitize_filename(result.plan_name)}.xlsx"
        output_path = _unique_output_path(OUTPUT_FOLDER, output_filename)

        write_workbook(result.pages, output_path)
        logger.info("Success -> %s", output_path)
        return True
    except Exception as exc:  # noqa: BLE001 - per-file failure boundary
        logger.error("Failed to process %s: %s", pdf_path, exc)
        return False


def _move_to_failed(path: str) -> None:
    os.makedirs(FAILED_FOLDER, exist_ok=True)
    dest_name = os.path.basename(path)
    dest_path = os.path.join(FAILED_FOLDER, dest_name)
    # If destination exists, create a unique filename to avoid FileExistsError
    if os.path.exists(dest_path):
        dest_path = _unique_output_path(FAILED_FOLDER, dest_name)

    try:
        shutil.move(path, dest_path)
    except PermissionError as exc:
        logger.error("Permission denied moving '%s' to failed folder: %s", path, exc)
    except Exception as exc:
        logger.error("Failed to move '%s' to failed folder: %s", path, exc)


def run() -> None:
    configure_logging()
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_paths = [
        os.path.join(INPUT_FOLDER, name)
        for name in sorted(os.listdir(INPUT_FOLDER))
        if name.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        logger.info("No PDF files found in %s. Exiting.", INPUT_FOLDER)
        return

    logger.info("Found %d PDF(s) to process.", len(pdf_paths))

    successes = 0
    for pdf_path in pdf_paths:
        if process_input_file(pdf_path):
            successes += 1
        else:
            _move_to_failed(pdf_path)

    logger.info("Done. %d/%d file(s) processed successfully.", successes, len(pdf_paths))


if __name__ == "__main__":
    run()
