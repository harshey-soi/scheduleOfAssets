from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import FOOTER_LINE_PATTERNS, ROW_Y_TOLERANCE
from models import ScheduleRow
from utils import cluster_by_position, is_numeric_value, join_text, strip_row_prefix

logger = logging.getLogger(__name__)

# =========================================================
# WORD MODEL
# =========================================================


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


# =========================================================
# GAP-DETECTION TUNING
# =========================================================
# These are NOT hardcoded page coordinates -- they define what counts as
# an unusually wide gap *relative to* the normal word spacing already
# observed on that same row, so the same logic self-calibrates across
# different PDFs, fonts, and font sizes.

# No gap smaller than this (in PDF points) is ever considered "major",
# regardless of how it compares to the row's own baseline spacing. This
# is a floor against noise on very tightly-kerned text, not a column
# coordinate.
_MIN_MAJOR_GAP_PT = 12.0

# A gap is "major" once it's at least this many times wider than the
# median of the normal (non-major) gaps already seen earlier in the row.
_GAP_MULTIPLIER = 3.0

# Leading per-row markers to strip before any gap analysis, e.g. "*",
# "**", "***", or a dagger symbol, with an optional trailing period.
_LEADING_MARKER_RE = re.compile(r"^[\*\u2020]{1,3}\.?$")

# A genuine "Total"/summary row (e.g. "Total assets held for investment
# purposes", "Total investments") is structurally always the LAST such
# occurrence of the word "total" in the table -- a total, by definition,
# sums everything printed above it, so it can never appear before all of
# the individual line items. This is why matching the LAST occurrence
# (rather than, say, only the first word of a line) is the more reliable
# signal: a fund name or wrapped description that happens to contain
# "total" (e.g. "Pimco Total Return", or a wrapped continuation line that
# starts with "Total Return Fund") will still be followed by the real
# summary row further down the table, so it never becomes the LAST match
# unless there genuinely is no separate total row on the page at all.
#
# \btotal\b is a whole-word match, so things like "Subtotal" don't
# accidentally match.
_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s*)?-?\s*\d+\s*-?$",
    re.IGNORECASE,
)

_FOOTNOTE_MARKER_RE = re.compile(r"^\(\d+\)$")


def is_bottom_junk_line(line: List[Word]) -> bool:
    text = line_text(line).strip()
    lower = text.lower()

    if not text:
        return True

    # Page numbers
    compact = " ".join(text.split())

    if _PAGE_NUMBER_RE.fullmatch(compact):
        return True

    # See accompanying notes...
    if lower.startswith("see accompanying"):
        return True

    if "independent auditors" in lower:
        return True

    # Footnote legends
    if (
    "party-in-interest" in lower
    or "party in interest" in lower or "pany-in-intercst" in lower 
    ):
        return True

    if "cost information" in lower:
        return True

    if "indicates a party" in lower:
        return True

    if is_footer(text):
        return True

    if is_standalone_legend_line(line):
        return True

    return False
# =========================================================
# BASIC HELPERS
# =========================================================


def normalize_words(raw_words) -> List[Word]:
    return [Word(*w[:5]) for w in raw_words if w[4].strip()]


def group_lines(words: List[Word]) -> List[List[Word]]:
    """Group words into visual rows purely by vertical position, then sort
    each row's words left to right."""
    if not words:
        return []
    ys = [w.yc for w in words]
    groups = cluster_by_position(ys, ROW_Y_TOLERANCE)

    lines = []
    for g in groups:
        row = [words[i] for i in g]
        row.sort(key=lambda w: w.x0)
        lines.append(row)

    lines.sort(key=lambda row: min(w.y0 for w in row))
    return lines


def line_text(line: List[Word]) -> str:
    return join_text(w.text for w in line)


def is_footer(text: str) -> bool:
    lower = text.lower()

    if any(p.search(text) for p in FOOTER_LINE_PATTERNS):
        return True
    if "cost information" in lower:
        return True
    if "participant-directed" in lower:
        return True
    return False


def is_standalone_legend_line(line: List[Word]) -> bool:
    """True for a standalone footnote/legend line (e.g. "* Party-in-
    interest to the Plan"), as opposed to a real table row that merely
    carries a "*"/"**" party-in-interest marker (e.g. "* MassMutual
    RetireSMART 2020 Fund").

    Both start with the same marker and are indistinguishable by that
    prefix alone. The reliable signal is that a real row always has a
    numeric value somewhere on it, while a standalone legend never does.
    """
    if not line:
        return False
    text = line_text(line)
    if not _LEADING_MARKER_RE.match(line[0].text.strip()):
        return False
    return not any(is_numeric_value(w.text) for w in line)


def is_row_numeric_only(words: List[Word]) -> bool:
    """Return True if the row contains only a numeric value, allowing a
    single leading dollar-sign token ("$") that may be separated from the
    digits by OCR/spacing. Examples considered numeric-only:

        ['$','27,154'] -> True
        ['2,458,218'] -> True
        ['Total','2,458,218'] -> False
    """
    if not words:
        return False
    texts = [w.text.strip() for w in words if w.text and w.text.strip()]
    if not texts:
        return False
    # Allow a leading isolated dollar sign
    if texts[0] == "$" and len(texts) >= 2:
        texts = texts[1:]
    return all(is_numeric_value(t) for t in texts)


# =========================================================
# HEADER DETECTION
# =========================================================


def detect_header(lines: List[List[Word]]) -> Optional[int]:
    header_idxs = []

    for i in range(min(25, len(lines))):
        text = line_text(lines[i]).lower()
        if (
            "identity of" in text
            or "issue" in text
            or "investment including" in text
            or "maturity date" in text
            or "current value" in text
        ):
            header_idxs.append(i)

    if not header_idxs:
        return None

    return min(header_idxs)


# =========================================================
# PER-ROW MARKER STRIPPING
# =========================================================


def strip_leading_markers(words: List[Word]) -> List[Word]:
    """Drop leading "*" / "**" / "***" / dagger marker tokens from the
    front of a row's word list, per the caller's instruction to ignore
    them entirely before doing anything else with the row.
    """
    idx = 0
    while idx < len(words) and _LEADING_MARKER_RE.match(words[idx].text.strip()):
        idx += 1
    return words[idx:]


# =========================================================
# FIRST-MAJOR-GAP IDENTITY SPLIT
# =========================================================


def find_identity_split_index(words: List[Word]) -> int:
    """Return how many leading words (after marker-stripping) belong to
    the Identity column, based on the FIRST unusually wide gap between
    consecutive words on this row.

    Walking left to right, this tracks the normal (non-major) word
    spacing seen so far on the row and flags a gap as "major" once it's
    both above the absolute noise floor (`_MIN_MAJOR_GAP_PT`) AND several
    times wider than that row's own typical spacing so far
    (`_GAP_MULTIPLIER`) -- e.g. in

        American Fund Ltd        Vanguard Mutual        4567   1234

    the small gaps between "American"/"Fund"/"Ltd" set the row's normal
    spacing baseline; the much wider gap before "Vanguard" is then
    recognized as the first major gap, so "American Fund Ltd" is
    Identity and everything from "Vanguard" onward is discarded.

    If no such gap exists on the row (e.g. a short, single-phrase line),
    this falls back to stopping at the first numeric-looking word, since
    numeric tokens never belong in the Identity text.
    """
    if len(words) <= 1:
        return len(words)

    gaps = [words[i + 1].x0 - words[i].x1 for i in range(len(words) - 1)]

    seen_gaps: List[float] = []
    for i, gap in enumerate(gaps):
        if seen_gaps:
            baseline = statistics.median(seen_gaps)
        else:
            baseline = _MIN_MAJOR_GAP_PT

        threshold = max(_MIN_MAJOR_GAP_PT, baseline * _GAP_MULTIPLIER)

        if gap >= threshold:
            return i + 1

        seen_gaps.append(gap)

    # No major gap found anywhere on the row -- fall back to stopping at
    # the first numeric token, since Identity text is never numeric.
    for i, w in enumerate(words):
        if is_numeric_value(w.text):
            return i

    return len(words)


# =========================================================
# LAST-NUMERIC-VALUE EXTRACTION
# =========================================================


def find_last_numeric_run(words: List[Word]) -> Optional[Tuple[int, int]]:
    """Return (start_index, end_index_inclusive) spanning the LAST run of
    numeric-looking tokens on the row that are only separated by small,
    non-major gaps -- i.e. tokens that the PDF/OCR text extractor split
    apart from what is really a single number (e.g. "1" and ",026,746"
    ending up as two separate word-boxes because of kerning around the
    thousands comma), as opposed to genuinely separate columns.

    A real second numeric column (e.g. Cost, printed well to the left of
    Current Value) is always separated from it by a wide/major gap -- the
    same kind of gap `find_identity_split_index` uses to find the
    Identity/Description boundary -- so this only ever merges fragments of
    one number back together; it does not merge two genuinely distinct
    column values into one.

    Returns None if the row has no numeric token at all.
    """
    last_idx = None
    for i in range(len(words) - 1, -1, -1):
        text = words[i].text.strip()

        # Ignore percentage values
        if "%" in text:
            continue

        # Ignore year ranges
        if re.fullmatch(r"\d{4}-\d{4}", text):
            continue
        if re.fullmatch(r"\(\d+\)", text.replace(" ", "")):
            continue
        if is_numeric_value(words[i].text):
            last_idx = i
            break
    if last_idx is None:
        return None

    start_idx = last_idx
    while start_idx > 0:
        prev = words[start_idx - 1]
        gap = words[start_idx].x0 - prev.x1
        if is_numeric_value(prev.text) and gap < _MIN_MAJOR_GAP_PT:
            start_idx -= 1
        else:
            break

    return start_idx, last_idx


def find_numeric_runs(words: List[Word]) -> List[Tuple[int, int]]:
    """Return all numeric runs on the row as (start_idx, end_idx) pairs,
    merging adjacent numeric tokens separated only by small gaps.
    Runs are returned left-to-right.
    """
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(words)
    while i < n:
        if is_numeric_value(words[i].text):
            start = i
            j = i
            while j + 1 < n:
                gap = words[j + 1].x0 - words[j].x1
                if is_numeric_value(words[j + 1].text) and gap < _MIN_MAJOR_GAP_PT:
                    j += 1
                else:
                    break
            runs.append((start, j))
            i = j + 1
        else:
            i += 1
    return runs


def clean_value_text(start_index: int, end_index: int, words: List[Word]) -> str:
    """Reassemble and clean the value spanning `words[start_index:end_index+1]`,
    restoring a leading "$" if the token immediately before the run
    indicates one -- including the OCR quirk where a lone "$" is misread
    as the digit "5".
    """
    raw = "".join(w.text for w in words[start_index : end_index + 1])
    matches = re.findall(r"[\d,]+(?:\.\d+)?", raw)
    value = "".join(matches) if matches else raw

    if start_index > 0:
        prev_text = words[start_index - 1].text.strip()
        if prev_text in ("$", "5"):
            value = "$ " + value

    return value


# =========================================================
# MAIN PARSER
# =========================================================


def parse_schedule_h_page(raw_words):
    words = normalize_words(raw_words)

    # Cut off the bottom 5% of the page to avoid footer text.
    if words:
        max_y = max(w.y1 for w in words)
        cutoff_y = max_y * 0.95
        words = [w for w in words if w.y0 < cutoff_y]

    lines = group_lines(words)

    start_idx = detect_header(lines)
    if start_idx is None:
        return [], False

    body_lines = lines[start_idx + 1 :]

    # Debug: show body lines before footer trimming when in DEBUG mode
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Body lines (pre-trim) count=%d", len(body_lines))
        for bi, bl in enumerate(body_lines):
            try:
                txt = line_text(bl).strip()
            except Exception:
                txt = "<unprintable>"
            jb = is_bottom_junk_line(bl)
            valhit = find_last_numeric_run(strip_leading_markers(bl))
            logger.debug("LINE %03d | bottom_junk=%s | value_hit=%s | text='%s'", bi, jb, valhit, txt)

    # ------------------------------------
    # Trim footer junk from bottom upward.
    # ------------------------------------

    last_content_idx = len(body_lines) - 1

    while last_content_idx >= 0:
        if is_bottom_junk_line(body_lines[last_content_idx]):
            last_content_idx -= 1
        else:
            break

    body_lines = body_lines[: last_content_idx + 1]

    # --------------------------------------------------
    # Detect approximate X position of the numeric (right) column
    # across the page so we prefer numeric runs that appear in
    # that right-hand area rather than numeric tokens embedded in
    # the identity/description text (e.g. '500' inside a name).
    # --------------------------------------------------
    numeric_x_candidates: List[float] = []
    for line in body_lines:
        row_words = strip_leading_markers(line)
        if not row_words:
            continue
        hit = find_last_numeric_run(row_words)
        if hit is None:
            continue
        s, e = hit
        # average center x of the numeric run
        coords = [w.xc for w in row_words[s : e + 1]]
        if coords:
            numeric_x_candidates.append(statistics.mean(coords))

    numeric_x_median: Optional[float]
    if numeric_x_candidates:
        numeric_x_median = statistics.median(numeric_x_candidates)
    else:
        numeric_x_median = None

    # How far left a run can be from the median and still be
    # considered to live in the numeric column. Tuned conservatively.
    _NUMERIC_COLUMN_TOLERANCE_PT = 30.0

    rows: List[ScheduleRow] = []
    current_identity: Optional[str] = None
    row_open = False
    can_merge_continuation = False
    previous_row_had_value = False
    last_completed_row_index = None
    open_row_has_extra_content = False

    for idx, line in enumerate(body_lines):
        text = line_text(line).strip()
        lower = text.lower()

        if len(text) < 5:
            continue

        # ------------------------------------
        # Skip wrapped header-continuation lines.
        # ------------------------------------
        if (
            "fl lessor" in lower
            or "if lessor" in lower
            or "similar party" in lower
            or "par, or maturity" in lower
            or "par, 0r maturity" in lower
            or "cost value" in lower
            or "maturity date, rate" in lower
        ):
            continue

        # ------------------------------------
        # Skip header / column-label text.
        # ------------------------------------
        if (
            "identity of issue" in lower
            or "borrower" in lower
            or "investment including" in lower
            or "interest, collateral" in lower
            or lower.startswith("maturity date")
            or lower.startswith("par, or maturity value")
            or "current value" in lower
            or "(a)" in lower
            or "(b)" in lower
            or "(c)" in lower
            or "(d)" in lower
            or "(e)" in lower
            or lower.startswith("identity of")
        ):
            continue

        # ------------------------------------
        # Skip footer / notes text.
        # ------------------------------------
        if (
            "indicates a party" in lower
            or is_footer(text)
        ):
            continue

        # ------------------------------------
        # Skip standalone legend lines (e.g. "* Party-in-interest to the
        # Plan"), while keeping real rows that merely carry a "*"
        # party-in-interest marker.
        # ------------------------------------
        if is_standalone_legend_line(line):
            continue

        # ------------------------------------
        # Strip the leading "*"/"**"/"***" marker(s), per row, before any
        # gap analysis.
        # ------------------------------------
        row_words = strip_leading_markers(line)
        if not row_words:
            continue

        compact_text = "".join(w.text for w in row_words)
        compact_text = compact_text.replace(" ", "")

        if re.fullmatch(r"\(\d+\)", compact_text):
            continue

        # ------------------------------------
        # Known Schedule H row type: normalize "Participants Loans" rows
        # to a clean identity regardless of gap spacing, since this is
        # standard Form 5500 terminology, not anything PDF-specific.
        # ------------------------------------
        # Normalize common OCR/typography variants for participant loans
        # (e.g. "Participant loans", "Participants Loans", "Participant Loan(s)").
        if re.search(r"\bparticipant(?:s)?\s+loans?\b", lower):
            value_hit = find_last_numeric_run(row_words)
            logger.info(
                "DEBUG | text='%s' | value_hit=%s | row_open=%s | current_identity=%s",
                text,
                value_hit,
                row_open,
                current_identity,
            )
            if row_open and current_identity:
                rows.append(ScheduleRow(identity=current_identity, description="", cost="", current_value=""))
            current_identity = "Participant loans"
            if value_hit is not None:
                start_idx, end_idx = value_hit
                value = clean_value_text(start_idx, end_idx, row_words)
                rows.append(
                    ScheduleRow(
                        identity=current_identity,
                        description="",
                        cost="",
                        current_value=value,
                    )
                )
                current_identity = None
                row_open = False
                can_merge_continuation = False
                open_row_has_extra_content = False
            else:
                row_open = True
            continue

        value_hit = find_last_numeric_run(row_words)

        # Choose which numeric run to use. Prefer the numeric run that is
        # visually farthest to the right (the current-value column). To do
        # that, enumerate all numeric runs on the row and pick the one with
        # the largest average X coordinate. This avoids accidentally picking
        # the left-side 'Cost' number when two numeric columns are present.
        if row_words:
            runs = find_numeric_runs(row_words)
            if runs:
                # Pick the run whose right edge is farthest to the right
                # (more robust than mean x when widths differ).
                def run_right_x(r):
                    s, e = r
                    return max(w.x1 for w in row_words[s : e + 1])

                best = max(runs, key=run_right_x)
                value_hit = best
            else:
                value_hit = None

        # Fallback: some values are printed on the page as a single large
        # number token that the numeric-run merging missed (e.g. long
        # totals like 8,732,964). If we didn't find a numeric run but the
        # row contains a large numeric token (>=5 digits), accept that
        # token as the value. This avoids mis-attributing small numbers
        # inside identities while capturing real totals.
        if value_hit is None:
            for i, w in enumerate(row_words):
                digits = re.sub(r"\D", "", w.text)
                if len(digits) >= 5:
                    logger.info(
                        "FALLBACK LARGE-NUM TOKEN -> using token '%s' as value on line: %s",
                        w.text,
                        line_text(row_words),
                    )
                    value_hit = (i, i)
                    break

        # Quick guard: handle cases where the row contains only one or more
        # isolated dollar-sign tokens followed by the numeric value (e.g.
        # "$ $ 10,857,894") so we don't try to treat those dollar tokens
        # as an Identity. Only apply this when the words immediately to the
        # left of the detected numeric run are *all* dollar signs; do NOT
        # use this for rows that contain descriptive text before the value
        # (e.g. "Ave Maria Bond Fund $ $ 167,691").
        if value_hit is not None:
            s_val, e_val = value_hit
            leading = row_words[:s_val]
            if leading and all(w.text.strip() in ("$",) for w in leading):
                logger.info(
                    "DOLLAR-LEADING VALUE LINE -> emitting value-only | full_row=%s",
                    line_text(row_words),
                )
                start_idx, end_idx = value_hit
                value = clean_value_text(start_idx, end_idx, row_words)
                rows.append(
                    ScheduleRow(identity="", description="", cost="", current_value=value)
                )
                previous_row_had_value = True
                last_completed_row_index = len(rows) - 1
                current_identity = None
                row_open = False
                can_merge_continuation = False
                open_row_has_extra_content = False
                continue

        if not row_open:
            # This line must start a new row.
            split_idx = find_identity_split_index(row_words)
            identity_words = row_words[:split_idx]
            identity_text = strip_row_prefix(join_text(w.text for w in identity_words)).strip()
            # If this line contains a numeric value on the right, prefer
            # to define the identity as all words that sit left of the
            # numeric run's left edge. This avoids pulling description
            # fragments (e.g. "Market Fund Select Class") into the
            # identity when they are visually between the identity and
            # the right-hand value column.
            if value_hit is not None:
                try:
                    s_val, e_val = value_hit
                    run_left = min(w.x0 for w in row_words[s_val : e_val + 1])
                    cutoff = run_left - (_MIN_MAJOR_GAP_PT / 2.0)
                    trimmed = [w for w in row_words if w.x1 < cutoff]
                    if trimmed and len(trimmed) < len(identity_words):
                        identity_words = trimmed
                        identity_text = strip_row_prefix(join_text(w.text for w in identity_words)).strip()
                except Exception:
                    pass
            # Footnote marker accidentally detected as identity
            if re.fullmatch(r"\(\s*\d+\s*\)", identity_text):
                continue
            # A lone "$" detected as identity is usually junk; however,
            # if the row contains a numeric value we should not skip it
            # here because it may be a dollar-leading value-only row
            # (e.g. "$ 29,367,921 $ 26,449,818"). Only skip when no
            # numeric run was detected.
            if identity_text.strip() == "$" and value_hit is None:
                continue

            # If the identity_text looks like a section header ending with
            # a colon or dash (e.g. "General account -"), emit it as its
            # own row and process the remainder of the line (if any) as a
            # separate identity/value pair. This handles cases where the
            # extractor put a header and the first item on the same PDF
            # text line.
            if re.search(r"[:\-]\s*$", identity_text):
                header_row = identity_text
                rows.append(
                    ScheduleRow(identity=header_row, description="", cost="", current_value="")
                )
                # Process remaining words on the same line after the header
                remainder = row_words[split_idx:]
                if not remainder:
                    continue
                # Attempt to find identity/value in the remainder
                rem_value_hit = find_last_numeric_run(remainder)
                rem_split = find_identity_split_index(remainder)
                rem_identity = strip_row_prefix(join_text(w.text for w in remainder[:rem_split])).strip()
                if rem_value_hit is not None:
                    s2, e2 = rem_value_hit
                    value2 = clean_value_text(s2, e2, remainder)
                    rows.append(
                        ScheduleRow(identity=rem_identity, description="", cost="", current_value=value2)
                    )
                    previous_row_had_value = True
                    last_completed_row_index = len(rows) - 1
                    continue
                else:
                    # Start a new open row using the remainder identity
                    if rem_identity:
                        current_identity = rem_identity
                        can_merge_continuation = (rem_split == len(remainder))
                        open_row_has_extra_content = (rem_split < len(remainder))
                        row_open = True
                    continue
            # If the entire row is numeric (e.g. a large total printed on
            # its own line), emit it as a value-only row with an empty
            # identity. This covers single-token numeric lines that would
            # otherwise be misinterpreted as an Identity due to the
            # gap-based split logic.
            # Special-case: sometimes the extractor yields one or more
            # isolated dollar-sign tokens before the numeric value (e.g.
            # "$ $ 10,857,894"). Those should be treated as a
            # value-only row too.
            if identity_words and all(w.text.strip() == "$" for w in identity_words):
                logger.info(
                    "DOLLAR-ONLY IDENTITY ROW -> treating as value-only | full_row=%s",
                    line_text(row_words),
                )
                # Prefer the already-detected numeric run (the rightmost
                # run) if available; otherwise fall back to scanning the
                # whole row. This avoids concatenating multiple numbers
                # (e.g. "29,367,921" + "26,449,818").
                if value_hit is not None:
                    start_idx, end_idx = value_hit
                else:
                    start_idx = 0
                    end_idx = len(row_words) - 1
                value = clean_value_text(start_idx, end_idx, row_words)
                rows.append(
                    ScheduleRow(
                        identity="",
                        description="",
                        cost="",
                        current_value=value,
                    )
                )
                previous_row_had_value = True
                last_completed_row_index = len(rows) - 1
                continue

            if identity_words and is_row_numeric_only(identity_words):
                logger.info(
                    "NUMERIC-ONLY IDENTITY ROW -> emitting value-only | full_row=%s",
                    line_text(row_words),
                )
                start_idx = 0
                end_idx = len(row_words) - 1
                value = clean_value_text(start_idx, end_idx, row_words)
                rows.append(
                    ScheduleRow(
                        identity="",
                        description="",
                        cost="",
                        current_value=value,
                    )
                )
                previous_row_had_value = True
                last_completed_row_index = len(rows) - 1
                continue

            # If there's no identity text but the row contains a
            # numeric run, treat this as a value-only line and emit
            # it as its own row (so it doesn't become the prefix of
            # the following identity).
            if not identity_text:
                if value_hit is not None:
                    start_idx, end_idx = value_hit
                    value = clean_value_text(start_idx, end_idx, row_words)
                    rows.append(
                        ScheduleRow(
                            identity="",
                            description="",
                            cost="",
                            current_value=value,
                        )
                    )
                    previous_row_had_value = True
                    last_completed_row_index = len(rows) - 1
                continue

            # Skip Mutual Funds section header
            if identity_text.strip().lower().startswith("mutual funds"):
                continue
            
            # Participant-loan descriptive lines
            if (
                "%" in identity_text
                or re.search(r"\d{4}-\d{4}", identity_text)
            ):
                continue

            logger.info(
                "STATE CHECK | row_open=%s | current_identity=%s",
                row_open,
                current_identity,
            )
            # Heuristic: if the previous visual line (one above in the
            # PDF) looks like a short identity with no numeric value, and
            # this current line contains a value, then the previous line
            # is likely the leading identity fragment and should be
            # merged with the current identity. This addresses cases
            # where a name wraps onto the next visual line that also
            # carries the class/fund text and value.
            if (
                idx > 0
                and identity_text
                and value_hit is not None
            ):
                prev_line = body_lines[idx - 1]
                prev_words = strip_leading_markers(prev_line)
                if prev_words:
                    prev_split = find_identity_split_index(prev_words)
                    prev_identity = strip_row_prefix(join_text(w.text for w in prev_words[:prev_split])).strip()
                    prev_has_numeric = any(is_numeric_value(w.text) for w in prev_words)
                    # Only merge when previous line is short, non-numeric,
                    # and not a section header.
                    if (
                        prev_identity
                        and not prev_has_numeric
                        and len(prev_identity.split()) <= 3
                        and not re.search(r"[:\-]\s*$", prev_identity)
                    ):
                        merged_identity = (prev_identity + " " + identity_text).strip()
                        start_idx, end_idx = value_hit
                        value = clean_value_text(start_idx, end_idx, row_words)
                        # If the previous line was already emitted as a
                        # standalone row with an empty value, replace it;
                        # otherwise append a new merged row.
                        if (
                            last_completed_row_index is not None
                            and rows
                            and rows[last_completed_row_index].identity == prev_identity
                            and not rows[last_completed_row_index].current_value
                        ):
                            rows[last_completed_row_index].identity = merged_identity
                            rows[last_completed_row_index].current_value = value
                        else:
                            rows.append(
                                ScheduleRow(identity=merged_identity, description="", cost="", current_value=value)
                            )
                            last_completed_row_index = len(rows) - 1
                        previous_row_had_value = True
                        current_identity = None
                        row_open = False
                        open_row_has_extra_content = False
                        can_merge_continuation = False
                        continue
            current_identity = identity_text
            # Was there anything after the identity split?
            # If yes, this row already contains extra content and
            # should not later absorb text from the next line.
            can_merge_continuation = (
                split_idx == len(row_words)
            )

            open_row_has_extra_content = (
                split_idx < len(row_words)
            )

            row_open = True
            logger.info(
                "OPENING ROW -> %s",
                current_identity,
            )

            logger.info(
                "ROW DEBUG | identity=%s | value_hit=%s | full_row=%s",
                identity_text,
                value_hit,
                text,
            )
            if value_hit is not None:

                start_idx, end_idx = value_hit
                value = clean_value_text(start_idx, end_idx, row_words)
                logger.info(
                    "CLOSING ROW -> identity=%s value=%s",
                    current_identity,
                    value,
                )
                rows.append(
                    ScheduleRow(
                        identity=current_identity,
                        description="",
                        cost="",
                        current_value=value,
                    )
                )
                previous_row_had_value = True
                last_completed_row_index = len(rows) - 1

                current_identity = None
                row_open = False
                open_row_has_extra_content = False
        else:

            
            # Continuation line of an already-open ro
            if value_hit is not None:
                
                # Accept any close textual variant of the participant-loans
                # identity when handling continuation lines.
                if current_identity and "participant" in current_identity.lower() and "loan" in current_identity.lower():
                    start_idx, end_idx = value_hit
                    value = clean_value_text(start_idx, end_idx, row_words)

                    logger.info(
                        "NO VALUE ROW | current_identity=%s | full_row=%s",
                        current_identity,
                        text,
                    )

                    rows.append(
                        ScheduleRow(
                            identity="Participants Loans",
                            description="",
                            cost="",
                            current_value=value,
                        )
                    )

                    previous_row_had_value = True
                    last_completed_row_index = len(rows) - 1

                    current_identity = None
                    row_open = False
                    can_merge_continuation = False
                    open_row_has_extra_content = False

                    continue

                split_idx_current = find_identity_split_index(row_words)
                current_identity_part = strip_row_prefix(
                    join_text(w.text for w in row_words[:split_idx_current])
                ).strip()
                if re.fullmatch(r"\(\s*\d+\s*\)", current_identity_part):
                    continue
                if (
                    not can_merge_continuation
                    and current_identity
                    and open_row_has_extra_content
                    and len(current_identity_part.split()) > 1
                ):

                    rows.append(
                        ScheduleRow(
                            identity=current_identity,
                            description="",
                            cost="",
                            current_value="",
                        )
                    )
                    previous_row_had_value = False

                    split_idx = find_identity_split_index(row_words)

                    merged_identity = strip_row_prefix(
                        join_text(w.text for w in row_words[:split_idx])
                    ).strip()

                else:
                    merged_identity = current_identity or ""

                if can_merge_continuation:
                    split_idx = find_identity_split_index(row_words)

                    continuation_text = strip_row_prefix(
                        join_text(w.text for w in row_words[:split_idx])
                    ).strip()

                    if continuation_text:
                        merged_identity = (
                            merged_identity + " " + continuation_text
                        ).strip()

                start_idx, end_idx = value_hit
                value = clean_value_text(start_idx, end_idx, row_words)

                rows.append(
                    ScheduleRow(
                        identity=merged_identity,
                        description="",
                        cost="",
                        current_value=value,
                    )
                )

                current_identity = None
                row_open = False
                can_merge_continuation = False
                open_row_has_extra_content = False

            else:
                split_idx = find_identity_split_index(row_words)

                identity_part = strip_row_prefix(
                    join_text(w.text for w in row_words[:split_idx])
                ).strip()

                # CASE 1:
                # abc
                # lmn ... 456
                #
                # true wrapped identity fragment
                if can_merge_continuation:
                    continue

                # CASE 2:
                # lmn pqr ** 456
                # fru
                # abc ... 4556
                #
                # single-word description fragment -> ignore
                if (
                    previous_row_had_value
                    and len(identity_part.split()) == 1
                    and last_completed_row_index is not None
                ):
                    rows[last_completed_row_index].identity = (
                        rows[last_completed_row_index].identity
                        + " "
                        + identity_part
                    ).strip()

                    continue

                # CASE 3:
                # def ipe **
                # ghi ... 456
                #
                # standalone no-value row
                rows.append(
                    ScheduleRow(
                        identity=current_identity or "",
                        description="",
                        cost="",
                        current_value="",
                    )
                )

                previous_row_had_value = False

                current_identity = identity_part

                can_merge_continuation = (
                    split_idx == len(row_words)
                )

                open_row_has_extra_content = (
                    split_idx < len(row_words)
                )

                row_open = True
    # A row that opened but never found a value (e.g. end of page reached
    # mid-row) is dropped rather than emitted with an empty value, since
    # there's nothing useful to report for it.

    # Emit any still-open row with blank value
    if row_open and current_identity:
        rows.append(
            ScheduleRow(
                identity=current_identity,
                description="",
                cost="",
                current_value="",
            )
        )

    return rows, False  # False: no Cost column in the simplified output
