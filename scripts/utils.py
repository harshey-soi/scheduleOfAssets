"""Small, stateless helper functions used across the pipeline.

Nothing in this module is aware of PDFs, pages, or coordinates -- it only
operates on plain strings and numbers, which keeps it trivially testable.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, List, Sequence

from config import NUMERIC_VALUE_RE, ROW_PREFIX_RE


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, for use by the CLI entry point."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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
