"""Small, stateless helper functions used across the pipeline.

Nothing in this module is aware of PDFs, pages, or coordinates -- it only
operates on plain strings and numbers, which keeps it trivially testable.
"""
from __future__ import annotations
import os
import time
import logging
import re
from typing import Iterable, List, Sequence

from config import NUMERIC_VALUE_RE, ROW_PREFIX_RE

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

def get_log_path() -> str:
    """One log file per day: logs/log_YYYY-MM-DD.log"""
    return os.path.join(LOG_DIR, f"log_{time.strftime('%Y-%m-%d')}.log")

_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


class _BufferHandler(logging.Handler):
    """Holds log records in memory for the file currently being processed."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


_buffer = _BufferHandler()
_short_lines: list = []


def configure_logging(level: int = logging.DEBUG) -> None:
    """Nothing is printed by logging. Records are buffered per file."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(_buffer)
    # Keep noisy third-party libraries out of the detailed log
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def step(message: str) -> None:
    """Print a progress line to the terminal and remember it for the log file."""
    print(message, flush=True)
    _short_lines.append(f"{time.strftime('%H:%M:%S')} | {message}")


def begin_file(pdf_path: str) -> None:
    """Reset buffers at the start of each PDF."""
    _buffer.records.clear()
    _short_lines.clear()


def end_file(pdf_path: str, success: bool) -> None:
    """Append this PDF's entry to today's log file. Short on success, detailed on failure."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = get_log_path()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if success else "FAILED"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {stamp} | {status} | {pdf_path} =====\n")
        for line in _short_lines:
            f.write(line + "\n")
        if not success:
            f.write("--- detailed log ---\n")
            for record in _buffer.records:
                f.write(_FORMATTER.format(record) + "\n")
    if not success:
        print("Error in processing file. Please check the log book: " + log_path, flush=True)


def sanitize_filename(name: str) -> str:
    """Remove characters that are illegal in Windows/Unix filenames."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "Unknown Plan Name"

def normalize_number_spacing(text: str) -> str:
    """Collapse stray whitespace introduced around thousands separators by
    text extraction/OCR, e.g. '245, 315' -> '245,315'.
    """
    if not text:
        return text
    return re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", text)


def is_numeric_value(text: str) -> bool:
    """True if the given text looks like a dollar-style numeric value."""
    if not text:
        return False
    return bool(NUMERIC_VALUE_RE.match(text.strip()))


def strip_row_prefix(text: str) -> str:
    """Remove leading list markers such as '*', '1.', '(a)', '(i)'."""
    if not text:
        return text
    return ROW_PREFIX_RE.sub("", text, count=1).strip()


def join_text(words: Iterable[str]) -> str:
    """Join word fragments with single spaces, collapsing extra whitespace."""
    joined = " ".join(w.strip() for w in words if w and w.strip())
    return re.sub(r"\s+", " ", joined).strip()


def cluster_by_position(values: Sequence[float], tolerance: float) -> List[List[int]]:
    """Cluster a sequence of 1-D positions (e.g. word Y-centers) into groups
    of indices, where members of a group are within `tolerance` of the
    group's running average position.

    This is the coordinate-based replacement for splitting on raw text
    lines: rows are discovered purely from vertical position, and columns
    (elsewhere) purely from horizontal position.

    Returns groups sorted by their mean position, ascending.
    """
    if not values:
        return []

    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: List[List[int]] = []
    group_means: List[float] = []

    for idx in order:
        pos = values[idx]
        placed = False
        # Because `order` is ascending, once a group's mean falls too far
        # behind `pos` it can never match again for any later item either,
        # so a simple linear scan here is both correct and cheap.
        for gi, mean in enumerate(group_means):
            if abs(pos - mean) <= tolerance:
                groups[gi].append(idx)
                n = len(groups[gi])
                group_means[gi] = mean + (pos - mean) / n
                placed = True
                break
        if not placed:
            groups.append([idx])
            group_means.append(pos)

    order_by_mean = sorted(range(len(groups)), key=lambda i: group_means[i])
    return [groups[i] for i in order_by_mean]
