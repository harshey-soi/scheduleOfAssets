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


def is_bottom_junk_line(line: List[Word]) -> bool:
    """Return True for footer, legend, or page-number lines that are not data rows."""
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
    """Convert raw word tuples into `Word` objects and drop blanks."""
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
    """Rebuild a human-readable text line from a list of `Word` objects."""
    return join_text(w.text for w in line)


def is_footer(text: str) -> bool:
    """Return True when a text line matches known Schedule H footer boilerplate."""
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
    """Locate the first row that looks like the Schedule H column header."""
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
        # Ignore standalone 4-digit year tokens (e.g. '2022') which are
        # commonly part of descriptive parentheticals and not monetary
        # values. However, do NOT ignore them when they look like a
        # monetary value (for example when preceded by a "$" token or
        # when they are tightly adjacent to other numeric tokens). This
        # reduces false-positives (accepting a year as the row value)
        # while preserving real $YYYY amounts when context indicates
        # they're values.
        if re.fullmatch(r"\d{4}", text):
            try:
                y = int(text)
                if 1900 <= y <= 2099:
                    # Check nearby context before skipping the 4-digit token.
                    # Accept it (do not skip) if it's clearly part of a value:
                    # - preceded by a dollar-sign token, or
                    # - adjacent to another numeric token with only a small gap.
                    prev_ok = False
                    if i > 0:
                        prev_text = words[i - 1].text.strip()
                        if prev_text in ("$",):
                            prev_ok = True
                        elif is_numeric_value(prev_text):
                            gap = words[i].x0 - words[i - 1].x1
                            if gap < _MIN_MAJOR_GAP_PT:
                                prev_ok = True

                    next_ok = False
                    if i + 1 < len(words):
                        next_text = words[i + 1].text.strip()
                        if is_numeric_value(next_text):
                            gap = words[i + 1].x0 - words[i].x1
                            if gap < _MIN_MAJOR_GAP_PT:
                                next_ok = True

                    if not prev_ok and not next_ok:
                        continue
            except Exception:
                pass
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
    """Parse one Schedule H page from word boxes into normalized row objects."""
    words = normalize_words(raw_words)

    # NOTE: We intentionally do NOT apply a blind "bottom N% of page"
    # cutoff here. Computing max_y from the words themselves means that
    # on densely packed tables (where the last row sits right at the
    # bottom margin), that last row's own y-position IS the max_y, so a
    # percentage-based cutoff silently strips its words before line
    # grouping even happens -- before any debug logging can show it.
    # Real footer/legend text is instead removed below, after line
    # grouping, using pattern-based detection (`is_bottom_junk_line`),
    # which is safe because it inspects actual line content rather than
    # a blind vertical percentage.
    lines = group_lines(words)

    start_idx = detect_header(lines)
    if start_idx is None:
        return [], False

    # Original behavior: proceed without logging header context

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
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("NUMERIC RUNS FOUND: %s", runs)
                    for r in runs:
                        s, e = r
                        run_text = " ".join(w.text for w in row_words[s : e + 1])
                        meanx = statistics.mean([w.xc for w in row_words[s : e + 1]])
                        gap_before = None
                        if s > 0:
                            gap_before = row_words[s].x0 - row_words[s - 1].x1
                        has_comma = any(
                            ("," in w.text) for w in row_words[s : e + 1]
                        )
                        has_dollar = any(w.text.strip().startswith("$") for w in row_words[s : e + 1])
                        in_column = (
                            numeric_x_median is not None
                            and abs(meanx - numeric_x_median) <= _NUMERIC_COLUMN_TOLERANCE_PT
                        )
                        try:
                            embedded_flag = is_run_embedded(r)
                        except Exception:
                            embedded_flag = None
                        logger.debug(
                            "CANDIDATE %s | text='%s' | meanx=%.1f | gap_before=%s | in_column=%s | comma=%s | dollar=%s | embedded=%s",
                            r,
                            run_text,
                            meanx,
                            None if gap_before is None else round(gap_before, 1),
                            in_column,
                            has_comma,
                            has_dollar,
                            embedded_flag,
                        )
                # Pick the run whose right edge is farthest to the right
                # (more robust than mean x when widths differ).
                def run_right_x(r):
                    s, e = r
                    return max(w.x1 for w in row_words[s : e + 1])
                # Mixed-priority selection:
                # 1) Runs containing a comma or a leading '$' (likely monetary)
                # 2) Runs in the page-level numeric column (near median)
                # 3) Runs separated by a major gap from preceding text
                # 4) Fallback to the rightmost run
                def run_has_comma_or_dollar(r):
                    s, e = r
                    for w in row_words[s : e + 1]:
                        t = w.text.strip()
                        if "," in t or t.startswith("$"):
                            return True
                    return False

                runs_comma = [r for r in runs if run_has_comma_or_dollar(r)]

                chosen = None
                # Prefer comma/$ runs that are also in the column or separated
                if runs_comma:
                    runs_pref = []
                    for r in runs_comma:
                        s, e = r
                        meanx = statistics.mean([w.xc for w in row_words[s : e + 1]])
                        in_column = numeric_x_median is not None and abs(meanx - numeric_x_median) <= _NUMERIC_COLUMN_TOLERANCE_PT
                        gap_before = row_words[s].x0 - row_words[s - 1].x1 if s > 0 else float('inf')
                        separated = gap_before >= _MIN_MAJOR_GAP_PT
                        if in_column or separated:
                            runs_pref.append(r)
                    if runs_pref:
                        chosen = max(runs_pref, key=run_right_x)
                    else:
                        chosen = max(runs_comma, key=run_right_x)

                # If none chosen yet, prefer runs in numeric column
                if chosen is None and numeric_x_median is not None:
                    runs_in_column = []
                    for r in runs:
                        s, e = r
                        coords = [w.xc for w in row_words[s : e + 1]]
                        if coords and abs(statistics.mean(coords) - numeric_x_median) <= _NUMERIC_COLUMN_TOLERANCE_PT:
                            runs_in_column.append(r)
                    if runs_in_column:
                        # prefer ones separated by a major gap
                        runs_sep = []
                        relax_tolerance = _NUMERIC_COLUMN_TOLERANCE_PT * 5
                        for r in runs_in_column:
                            s, e = r
                            gap_before = row_words[s].x0 - row_words[s - 1].x1 if s > 0 else float('inf')
                            # require the run to also be reasonably close to the
                            # numeric column median when numeric_x_median is set;
                            # this prevents left-of-column runs (like '500' in
                            # '500 Index Fund') from being treated as separated
                            # numeric columns.
                            meanx = statistics.mean([w.xc for w in row_words[s : e + 1]])
                            if numeric_x_median is not None and abs(meanx - numeric_x_median) > relax_tolerance:
                                continue
                            if gap_before >= _MIN_MAJOR_GAP_PT:
                                runs_sep.append(r)
                        if runs_sep:
                            chosen = max(runs_sep, key=run_right_x)
                        else:
                            chosen = max(runs_in_column, key=run_right_x)

                # If still none, prefer runs separated by major gap
                if chosen is None:
                    runs_separated = []
                    relax_tolerance = _NUMERIC_COLUMN_TOLERANCE_PT * 5
                    for r in runs:
                        s, e = r
                        if s > 0:
                            gap_before = row_words[s].x0 - row_words[s - 1].x1
                            meanx = statistics.mean([w.xc for w in row_words[s : e + 1]])
                            if numeric_x_median is not None and abs(meanx - numeric_x_median) > relax_tolerance:
                                # skip separated runs that are far left of the
                                # numeric column median (likely part of identity)
                                continue
                            if gap_before >= _MIN_MAJOR_GAP_PT:
                                runs_separated.append(r)
                    if runs_separated:
                        chosen = max(runs_separated, key=run_right_x)
                    else:
                        chosen = max(runs, key=run_right_x)

                best = chosen

                # Post-filter: avoid selecting numeric runs that are clearly
                # embedded in the identity/description text (e.g. fund-year
                # tokens like "2060 TD"). A run is considered embedded if
                # it is tightly adjacent to alphabetic tokens on either side
                # (small gap) which indicates it's part of the name, not a
                # monetary column. If the chosen run appears embedded and
                # there is an alternative run that is not embedded, prefer
                # that alternative.
                def is_alpha_word(w):
                    return bool(re.search(r"[A-Za-z]", w.text))

                def is_run_embedded(r):
                    s, e = r
                    # gap to previous word
                    if s > 0:
                        gap_before = row_words[s].x0 - row_words[s - 1].x1
                        if gap_before < _MIN_MAJOR_GAP_PT and is_alpha_word(row_words[s - 1]):
                            return True
                    # gap to next word
                    if e + 1 < len(row_words):
                        gap_after = row_words[e + 1].x0 - row_words[e].x1
                        # Short alphabetic suffixes like 'TD' commonly follow
                        # fund-year tokens (e.g. '2060 TD') and indicate the
                        # numeric token is part of the identity even if the
                        # spacing is large. Treat short alpha-only tokens
                        # (1-3 letters) after the run as embedded context.
                        next_word = row_words[e + 1]
                        if (gap_after < _MIN_MAJOR_GAP_PT and is_alpha_word(next_word)):
                            return True
                        next_text = next_word.text.strip()
                        if re.fullmatch(r"[A-Za-z]{1,3}\.?", next_text):
                            return True
                    return False

                if best is not None and is_run_embedded(best):
                    # find an alternative non-embedded run among the previously
                    # considered candidate sets in priority order
                    alt = None
                    candidate_lists = [
                        runs_comma if 'runs_comma' in locals() else [],
                        (runs_in_column if 'runs_in_column' in locals() else []),
                        (runs_separated if 'runs_separated' in locals() else []),
                        runs,
                    ]
                    for clist in candidate_lists:
                        for r in sorted(clist, key=run_right_x, reverse=True):
                            if not is_run_embedded(r):
                                alt = r
                                break
                        if alt is not None:
                            break
                    if alt is not None:
                        best = alt
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

        # Heuristic: if the chosen value run is a short 4-digit token with
        # no comma and no leading '$', and it appears embedded in the
        # identity (or is followed by a short alphabetic suffix like
        # 'TD'), then treat it as NOT a value so that a following
        # numeric-only line (e.g. '41,219') can be used as the real value.
        if value_hit is not None:
            s_val, e_val = value_hit
            run_words = row_words[s_val : e_val + 1]
            run_text = "".join(w.text for w in run_words)
            digits = re.sub(r"\D", "", run_text)
            has_comma_or_dollar = any(("," in w.text or w.text.strip().startswith("$")) for w in run_words)
            # small 4-digit token without monetary cues
            if len(digits) == 4 and not has_comma_or_dollar:
                # check adjacency to alpha words (embedded) or short alpha suffix
                embedded = False
                if s_val > 0:
                    gap_before = row_words[s_val].x0 - row_words[s_val - 1].x1
                    if gap_before < _MIN_MAJOR_GAP_PT and re.search(r"[A-Za-z]", row_words[s_val - 1].text):
                        embedded = True
                if e_val + 1 < len(row_words):
                    gap_after = row_words[e_val + 1].x0 - row_words[e_val].x1
                    next_text = row_words[e_val + 1].text.strip()
                    if gap_after < _MIN_MAJOR_GAP_PT and re.search(r"[A-Za-z]", row_words[e_val + 1].text):
                        embedded = True
                    if re.fullmatch(r"[A-Za-z]{1,3}\.?", next_text):
                        embedded = True
                if embedded:
                    value_hit = None
                # Allow 4-digit numeric tokens when they look like a value:
                # - preceded by a dollar-sign token, or
                # - positioned near the page's detected numeric column median.
                if len(digits) == 4:
                    # Use the run's start index (`s_val`) rather than a
                    # loop variable that may not be defined in this scope.
                    prev_text = row_words[s_val - 1].text.strip() if s_val > 0 else ""
                    is_prev_dollar = prev_text in ("$",)
                    is_in_numeric_column = False
                    if numeric_x_median is not None:
                        try:
                            if abs(row_words[s_val].xc - numeric_x_median) < _NUMERIC_COLUMN_TOLERANCE_PT:
                                is_in_numeric_column = True
                        except Exception:
                            is_in_numeric_column = False

                    if is_prev_dollar or is_in_numeric_column:
                        logger.info(
                            "FALLBACK 4-DIGIT TOKEN -> using token '%s' as value on line: %s",
                            run_text,
                            line_text(row_words),
                        )
                        value_hit = (s_val, e_val)

        # Additional heuristic: if the chosen run is a short numeric token
        # (no comma, no leading '$', digits <= 4) and the *next* visual
        # row is numeric-only (a separate line containing the real value),
        # prefer the next row instead of this short token which is likely
        # part of the identity (e.g. '500' in '500 Index Fund'). Clear
        # value_hit so the following numeric-only row will be treated as
        # the value for the currently-open identity.
        if value_hit is not None:
            try:
                s_val, e_val = value_hit
                run_words = row_words[s_val : e_val + 1]
                run_text = "".join(w.text for w in run_words)
                digits = re.sub(r"\D", "", run_text)
                has_comma_or_dollar = any(("," in w.text or w.text.strip().startswith("$")) for w in run_words)
                # Only treat short runs of exactly 4 digits as potential
                # embedded-year/fund tokens to reject; allow 1-3 digit
                # numeric values to be considered valid monetary values.
                if digits and len(digits) == 4 and not has_comma_or_dollar:
                    if idx + 1 < len(body_lines) and numeric_x_median is not None:
                        next_row = strip_leading_markers(body_lines[idx + 1])
                        # Require the next row to be numeric-only and its
                        # numeric run to sit in the page numeric column (to
                        # avoid grabbing a distant grand-total/subtotal).
                        if is_row_numeric_only(next_row):
                            next_hit = find_last_numeric_run(next_row)
                            if next_hit is not None:
                                ns, ne = next_hit
                                try:
                                    meanx_next = statistics.mean([w.xc for w in next_row[ns:ne+1]])
                                except Exception:
                                    meanx_next = None
                                # vertical proximity: next row should be close
                                # to current line (not several paragraphs below).
                                try:
                                    y_gap = next_row[0].yc - row_words[-1].yc
                                except Exception:
                                    y_gap = None
                                close_vertically = (y_gap is None) or (y_gap <= ROW_Y_TOLERANCE * 2)
                                in_numeric_column = meanx_next is not None and abs(meanx_next - numeric_x_median) <= _NUMERIC_COLUMN_TOLERANCE_PT
                                if in_numeric_column and close_vertically:
                                    logger.info(
                                        "PREFER NEXT ROW VALUE -> rejecting short run '%s' in favor of next numeric-only row: %s",
                                        run_text,
                                        line_text(next_row),
                                    )
                                    value_hit = None
            except Exception:
                pass

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
            # Skip obvious year-range headers always (e.g. "2019-2020").
            if re.search(r"\d{4}-\d{4}", identity_text):
                continue
            # Allow percent signs within identity when the row contains
            # a numeric value (e.g. "85% Fund ... 70,916"). Only skip
            # percent-containing identities when there's no detected
            # numeric value on the same row.
            if "%" in identity_text and value_hit is None:
                continue

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
