"""Configuration constants for the Schedule H extraction pipeline.

Everything in this file is a *generic* tuning parameter or pattern used to
recognize structure (headers, footers, numbers). Nothing here is specific
to any individual plan, filing, or PDF -- the same values are used across
every document the pipeline processes.
"""
from __future__ import annotations

import os
import re
import pytesseract
import shutil

# --------------------------------------------------------------------------
# Folder layout
# --------------------------------------------------------------------------
BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FOLDER = os.path.join(BASE_FOLDER, "input")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "output")
FAILED_FOLDER = os.path.join(OUTPUT_FOLDER, "failed")

# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------
# Rendering resolution used when a page must be rasterized for OCR, expressed
# as a PyMuPDF zoom factor (PDF points are 72/inch, so zoom = dpi / 72).
QUICK_OCR_ZOOM = 150 / 72   # cheap pass, used only to *detect* Schedule H pages
OCR_ZOOM = 300 / 72         # thorough pass, used to actually *extract* words

OCR_TESSERACT_CONFIG = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"

# Optional: allow users to specify the full tesseract binary path via
# several possible environment variable names (some teams use
# nonstandard names like `tesseract_prefix`). Accept either a full
# path to the exe or a directory path containing `tesseract.exe`.
_env_keys = [
    "TESSERACT_CMD",
    "TESSERACT_PATH",
    "TESSERACT_PREFIX",
    "tesseract_prefix",
    "TESSERACT",
]
TESSERACT_CMD = None
_used_env_key = None
for _k in _env_keys:
    _v = os.environ.get(_k)
    if not _v:
        continue
    # strip quotes and whitespace that some shells/IDEs include
    _v = _v.strip().strip('"').strip("'")
    # if user provided a directory, append the exe name
    if os.path.isdir(_v):
        _candidate = os.path.join(_v, "tesseract.exe")
    else:
        _candidate = _v
    if os.path.exists(_candidate):
        TESSERACT_CMD = _candidate
        _used_env_key = _k
        break

if not TESSERACT_CMD:
    _win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win_default):
        TESSERACT_CMD = _win_default

if TESSERACT_CMD:
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    except Exception:
        # Fail silently here; callers will raise a clear error when OCR is attempted
        pass

# Diagnostic: print resolved values so users see which tesseract binary
# the process will attempt to use (helps debug PATH vs env-var issues).
try:
    _which = shutil.which("tesseract")
except Exception:
    _which = None

if TESSERACT_CMD:
    print(f"[config] TESSERACT_CMD={TESSERACT_CMD!r}, exists={os.path.exists(TESSERACT_CMD)}, shutil.which('tesseract')={_which!r}")
else:
    print(f"[config] TESSERACT_CMD not set, shutil.which('tesseract')={_which!r}")

# A page is considered "not searchable" (i.e. requires OCR) once its native
# extractable text falls below this many characters.
MIN_SEARCHABLE_TEXT_CHARS = 40

# --------------------------------------------------------------------------
# Row / column grouping tolerances
# --------------------------------------------------------------------------
# Words are considered to be on the same visual row when the vertical
# distance between their vertical centers is within this many PDF points.
ROW_Y_TOLERANCE = 8.0

# --------------------------------------------------------------------------
# Regex patterns
# --------------------------------------------------------------------------
# Schedule H heading variants (case-insensitive). Used only to *locate*
# pages -- never to hardcode a specific filing's exact wording.
SCHEDULE_H_HEADING_RE = re.compile(
    r"schedule\s*h\b"
    r"|schedule\s*of\s*assets\s*\(\s*held\s*at\s*end\s*of\s*year\s*\)"
    r"|schedule\s*h\s*\(?\s*line\s*4i\s*\)?",
    re.IGNORECASE,
)

# Broader match used as a fallback for headings that simply say
# "Schedule of Assets" (many filings use this shorter heading).
# NOTE: removal: 'Schedule of Assets' fallback regex reverted to avoid
# overbroad page-matching. Use `SCHEDULE_H_HEADING_RE` for page detection.

# A different lettered schedule heading (e.g. "Schedule G") signals that
# Schedule H content has ended.
OTHER_SCHEDULE_HEADING_RE = re.compile(r"^\s*schedule\s*(?:of\s*)?([a-z])\b", re.IGNORECASE)

# Column header tokens. Each maps a canonical column to a regex matching the
# header text used for that column, across filings/layouts.
IDENTITY_HEADER_RE = re.compile(r"identity\s+of\s+issuer|identity\b", re.IGNORECASE)
DESCRIPTION_HEADER_RE = re.compile(r"description\s+of\s+investment|description\b", re.IGNORECASE)
COST_HEADER_RE = re.compile(r"\bcost\b", re.IGNORECASE)
VALUE_HEADER_RE = re.compile(r"current\s+value|\bvalue\b", re.IGNORECASE)

# Footer / boilerplate lines that mark the end of the data table on a page.
FOOTER_LINE_PATTERNS = [
    re.compile(r"party[\s-]?in[\s-]?interest", re.IGNORECASE),
    re.compile(r"column\s+\(?d\)?\s+is\s+not\s+applicable", re.IGNORECASE),
    re.compile(r"see\s+independent\s+auditor", re.IGNORECASE),
    re.compile(r"notes?\s+to\s+(the\s+)?financial\s+statements", re.IGNORECASE),
    re.compile(r"cost\s+information\s+(is\s+)?omitted", re.IGNORECASE),
    re.compile(r"accompanying\s+(independent\s+auditor.?s\s+)?report", re.IGNORECASE),
    re.compile(r"assets\s+(are|held)\s+.*participant[-\s]?directed", re.IGNORECASE),
    re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$"),  # page numbers only
]

# Row-start prefixes to strip from the beginning of an Identity cell, e.g.
# "*", "**", "1.", "(a)", "(i)".
ROW_PREFIX_RE = re.compile(
    r"^\s*(\*{1,3}|\d+\.|\([a-hj-z]\)|\([ivxlc]{1,4}\))\s*",
    re.IGNORECASE,
)

# Matches things that look like a dollar figure, e.g. "245,315",
# "1,245,118", "5,200", "(1,200)", "5,200.00", "$5,200"
NUMERIC_VALUE_RE = re.compile(r"^\(?\$?\s*-?[\d,]+(\.\d{1,2})?\)?\*{0,3}$")
