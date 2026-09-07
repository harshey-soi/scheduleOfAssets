# Schedule of Assets Extractor

Converts Form 5500 **Schedule H (Schedule of Assets Held at End of Year)** pages from PDF filings into structured Excel workbooks. Each PDF in `input/` is scanned for Schedule H pages, parsed using whichever layout it matches (normal text-based table, visual grid, or "tickered"/rotated layout), and written to a workbook in `output/` named after the plan.

## How it works

1. Every PDF in `input/` is opened and scanned for pages containing a Schedule H heading (`find_schedule_h_pages`).
2. Each matching page is classified into one of three layouts:
   - **Tickered** — rotated/ticker-style tables, requires OCR (`tickered.py`).
   - **Grid** — visually ruled tables detected via line/box analysis (`grid_extractor.py`).
   - **Normal** — standard text-extractable table (`schedule_parser.py`).
3. Rows are parsed into identity, description, cost (if present), and current value columns.
4. Results are written to an `.xlsx` workbook (`excel_writer.py`), one worksheet per page.
5. On success, the source PDF is deleted from `input/`. If extraction fails, an error workbook containing the traceback is written to `output/` instead, and the input PDF is left in place (or archived to `output/failed/` if it's locked).

## Project structure

```
input/               PDFs to be processed (drop files here)
output/               Generated Excel workbooks (and output/failed/ for locked input archives)
scripts/
  main.py             Entry point; orchestrates per-PDF processing
  config.py           Folder paths, OCR settings, regex patterns, tuning constants
  pdf_processor.py     PDF opening, page discovery, orientation, text/word extraction
  schedule_parser.py   Parses normal (text-based) Schedule H table layouts
  grid_extractor.py    Detects and parses visually gridded table layouts
  tickered.py          Detects and parses tickered/rotated table layouts (OCR-based)
  ocr_preprocessing.py Image preprocessing to improve OCR accuracy
  excel_writer.py       Writes parsed rows / error tracebacks to .xlsx workbooks
  models.py             Data classes for extraction results
  utils.py              Logging setup, filename sanitization, misc helpers
  run_debug.py          Runs extraction on a single PDF with verbose (DEBUG) logging
```

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and available, either on `PATH` or pointed to via an environment variable (`TESSERACT_CMD`, `TESSERACT_PATH`, or `TESSERACT_PREFIX`). On Windows, it also falls back to `C:\Program Files\Tesseract-OCR\tesseract.exe` if present.

Python dependencies (see [requirements.txt](requirements.txt)):

- `PyMuPDF` — PDF parsing/rendering
- `pandas`, `openpyxl` — Excel output
- `opencv-python`, `numpy`, `Pillow` — image preprocessing for OCR
- `pytesseract` — OCR engine bindings

## Setup

```powershell
pip install -r requirements.txt
```

Install Tesseract separately (it is not a Python package) and confirm it's discoverable:

```powershell
tesseract --version
```

## Usage

1. Place one or more Form 5500 PDF filings into `input/`.
2. Run the pipeline from the `scripts/` folder:

```powershell
cd scripts
python main.py
```

3. Processed workbooks appear in `output/`, named after each input PDF. Successfully processed PDFs are removed from `input/`; failed ones are left in place alongside an error workbook explaining the failure.

### Debugging a single file

To run one PDF with verbose logging (useful when diagnosing a parsing failure):

```powershell
cd scripts
python run_debug.py "..\input\example.pdf"
```

If no path is given, it picks the first PDF found in `input/`.
