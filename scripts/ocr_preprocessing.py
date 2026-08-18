"""Image preprocessing to improve OCR accuracy on scanned Schedule H pages."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Denoise and remove ruling lines from a scanned table image so OCR
    reads cell text more reliably.

    This is a purely visual cleanup pass -- it never touches text content,
    only pixels, so it is safe to apply uniformly to any scanned page
    regardless of its layout.
    """
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect and erase long horizontal and vertical ruling lines, which
    # otherwise fragment words and confuse OCR word-boxing.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    for line_mask in (horizontal_lines, vertical_lines):
        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(gray, [contour], -1, (255, 255, 255), 5)

    return Image.fromarray(gray)
