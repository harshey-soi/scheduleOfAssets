"""Entry point: convert every PDF in the input folder into a Schedule H
Excel workbook (named after the plan) in the output folder.

Usage:
    python main.py
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import List

from config import FAILED_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER
from excel_writer import write_workbook
from models import ExtractionResult, SchedulePageResult
from pdf_processor import (
    extract_plan_name,
    extract_words,
    find_schedule_h_pages,
    get_tickered_page_indices,
    normalize_orientation,
    open_document,
    is_tickered_document,
    is_tickered_among_pages,
)
import tickered
import grid_extractor
from schedule_parser import parse_schedule_h_page
from utils import configure_logging, sanitize_filename
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

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
                tickered_pages = get_tickered_page_indices(doc, target_pages) if target_pages else []
                grid_pages = grid_extractor.get_grid_page_indices(
                    doc,
                    [p for p in target_pages if p not in set(tickered_pages)],
                ) if target_pages else []
                normal_pages = [p for p in target_pages if p not in set(tickered_pages) and p not in set(grid_pages)]

                logger.info(
                    "Schedule H page classification | tickered=%s | grid=%s | normal=%s",
                    [p + 1 for p in tickered_pages],
                    [p + 1 for p in grid_pages],
                    [p + 1 for p in normal_pages],
                )

                tickered_among = bool(target_pages) and is_tickered_among_pages(doc, target_pages)
                tickered_doc = False
                try:
                    # Document-level detection can pick up header variants that
                    # aren't present on the exact Schedule H page slice. Use as
                    # a tolerant fallback when page-level check fails.
                    if not tickered_among:
                        tickered_doc = is_tickered_document(doc)
                except Exception:
                    tickered_doc = False

                if tickered_among or tickered_doc:
                    if tickered_among:
                        logger.info("Detected tickered layout on Schedule H pages: %s", pdf_path)
                    else:
                        logger.info("Detected tickered layout (document-level fallback): %s", pdf_path)
                    result = tickered.process_tickered_pdf(pdf_path)
                    # If tickered detection was a false positive (empty result),
                    # fall back to grid or standard pipeline rather than fail.
                    if result.is_empty():
                        logger.warning(
                            "Tickered pipeline produced no data for %s — falling back to alternative parsers",
                            pdf_path,
                        )
                        # Try grid extractor next
                        try:
                            if grid_pages:
                                logger.info(
                                    "Fallback: detected visual grid layout on Schedule H pages %s for %s",
                                    [p + 1 for p in grid_pages],
                                    pdf_path,
                                )
                                result = grid_extractor.process_grid_pdf(pdf_path, page_indices=grid_pages)
                        except Exception:
                            logger.debug("Grid detection/extraction fallback failed or raised an error")

                        # If still empty, try the default parser
                        if result.is_empty():
                            logger.info("Fallback: running standard Schedule H parser for %s", pdf_path)
                            result = process_pdf(pdf_path)
                elif grid_pages:
                    logger.info(
                        "Detected visual grid table layout on Schedule H pages %s: %s",
                        [p + 1 for p in grid_pages],
                        pdf_path,
                    )
                    result = grid_extractor.process_grid_pdf(pdf_path, page_indices=grid_pages)
                else:
                    result = process_pdf(pdf_path)
        finally:
            doc.close()

        if result.is_empty():
            raise ValueError("No data extracted from any Schedule H page.")

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        # Use the input PDF filename (without extension) for the output
        # workbook name rather than the extracted plan name.
        input_base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_filename = f"{sanitize_filename(input_base)}.xlsx"
        output_path = _unique_output_path(OUTPUT_FOLDER, output_filename)

        write_workbook(result.pages, output_path)
        logger.info("Success -> %s", output_path)
        # Delete the input PDF now that processing succeeded. If deletion
        # fails due to transient locks (Antivirus, Explorer preview), retry
        # a few times with exponential backoff. If still failing, move the
        # file to the failed folder as an archive fallback.
        delete_attempts = 5
        delay = 0.5
        deleted = False
        for attempt in range(1, delete_attempts + 1):
            try:
                # Diagnostic: log file existence and attributes before remove
                try:
                    exists = os.path.exists(pdf_path)
                    isfile = os.path.isfile(pdf_path)
                    size = os.path.getsize(pdf_path) if exists else -1
                    readable = os.access(pdf_path, os.R_OK)
                    writable = os.access(pdf_path, os.W_OK)
                    logger.debug(
                        "Delete attempt %d: exists=%s isfile=%s size=%d readable=%s writable=%s",
                        attempt,
                        exists,
                        isfile,
                        size,
                        readable,
                        writable,
                    )
                except Exception as _log_exc:
                    logger.debug("Failed to stat file before delete: %s", _log_exc)

                os.remove(pdf_path)
                logger.info("Deleted input file: %s", pdf_path)
                deleted = True
                break
            except PermissionError as exc:
                logger.warning(
                    "Permission denied deleting '%s' (attempt %d/%d): %s",
                    pdf_path,
                    attempt,
                    delete_attempts,
                    exc,
                )
                # Try clearing read-only bit on Windows as a remediation step
                try:
                    os.chmod(pdf_path, 0o666)
                    logger.info("Cleared read-only bit for %s", pdf_path)
                except Exception as chmod_exc:
                    logger.debug("Failed to chmod %s: %s", pdf_path, chmod_exc)

                time.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error("Failed to delete input file %s: %s", pdf_path, exc)
                break

        if not deleted:
            try:
                logger.info("Attempting to archive locked input to failed folder: %s", pdf_path)
                _move_to_failed(pdf_path)
            except Exception as exc:
                logger.error("Failed to archive locked input %s: %s", pdf_path, exc)
        return True
    except Exception as exc:  # noqa: BLE001 - per-file failure boundary
        import traceback
        tb = traceback.format_exc()
        logger.error("Failed to process %s: %s", pdf_path, exc)

        # Instead of moving the PDF to a failed folder, create an Excel
        # workbook with the same base name that contains the exception
        # traceback so users can inspect why processing failed.
        try:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            input_base = os.path.splitext(os.path.basename(pdf_path))[0]
            output_filename = f"{sanitize_filename(input_base)}.xlsx"
            output_path = _unique_output_path(OUTPUT_FOLDER, output_filename)
            from excel_writer import write_error_workbook

            write_error_workbook(output_path, tb)
            logger.info("Wrote error workbook -> %s", output_path)
        except Exception as write_exc:
            logger.error("Failed to write error workbook for %s: %s", pdf_path, write_exc)

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
            logger.info("Left failed input in place: %s", pdf_path)

    logger.info("Done. %d/%d file(s) processed successfully.", successes, len(pdf_paths))


if __name__ == "__main__":
    run()
