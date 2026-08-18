from __future__ import annotations

import logging
import sys
from pathlib import Path

from utils import configure_logging
from main import process_input_file


def main(argv: list[str]) -> int:
    configure_logging(logging.DEBUG)
    if len(argv) >= 2:
        pdf_path = argv[1]
    else:
        # pick first PDF in input/ if none provided
        inp = Path(__file__).resolve().parents[1] / "input"
        pdfs = sorted([p for p in inp.iterdir() if p.suffix.lower() == ".pdf"])
        if not pdfs:
            print("No PDF found in input/; provide a path argument")
            return 2
        pdf_path = str(pdfs[0])

    try:
        print(f"Running debug extraction on: {pdf_path}")
        ok = process_input_file(pdf_path)
        print("Done. Success:", ok)
        return 0 if ok else 1
    except Exception as e:
        logging.exception("Error during debug run")
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
