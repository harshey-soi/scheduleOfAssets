"""Image preprocessing to improve OCR accuracy on scanned Schedule H pages."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from config import DESKEW_MAX_ANGLE_DEG, DESKEW_MIN_ANGLE_DEG

logger = logging.getLogger(__name__)

# Width the page is shrunk to before angle search; full resolution adds cost
# without improving the estimate.
_SKEW_ANALYSIS_WIDTH = 800


def _binarize_for_skew(image: Image.Image) -> np.ndarray:
    """Return a downscaled binary image with text pixels set to 1."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    height, width = gray.shape[:2]
    if width > _SKEW_ANALYSIS_WIDTH:
        scale = _SKEW_ANALYSIS_WIDTH / float(width)
        gray = cv2.resize(
            gray,
            (_SKEW_ANALYSIS_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return binary.astype(np.uint8)


def _rotate_binary(binary: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the binary analysis image about its centre."""
    height, width = binary.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _skew_score(binary: np.ndarray, angle: float) -> float:
    """Score an angle by how sharply text separates into horizontal rows.

    When rows are level, each text line lands in a few image rows and the
    row-sum profile spikes; a tilted page spreads the same ink across many
    rows and flattens the profile.
    """
    rotated = _rotate_binary(binary, angle) if angle else binary
    profile = rotated.sum(axis=1, dtype=np.float64)
    deltas = np.diff(profile)
    return float(np.sum(deltas * deltas))


def estimate_skew_angle(image: Image.Image) -> float:
    """Return the page's slant in degrees (positive = counter-clockwise fix)."""
    binary = _binarize_for_skew(image)
    if int(binary.sum()) == 0:
        return 0.0

    limit = float(DESKEW_MAX_ANGLE_DEG)

    best_angle = 0.0
    best_score = _skew_score(binary, 0.0)
    for angle in np.arange(-limit, limit + 0.5, 0.5):
        score = _skew_score(binary, float(angle))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    # Refine around the coarse winner so sub-degree tilts are corrected too.
    for angle in np.arange(best_angle - 0.5, best_angle + 0.5 + 0.1, 0.1):
        if abs(angle) > limit:
            continue
        score = _skew_score(binary, float(angle))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return round(best_angle, 2)


def deskew_image(image: Image.Image) -> Image.Image:
    """Straighten a slightly tilted scan so table rows become horizontal.

    Pages that are already level are returned unchanged.
    """
    try:
        angle = estimate_skew_angle(image)
    except Exception as exc:
        logger.debug("Skew estimation failed: %s", exc)
        return image

    if abs(angle) < DESKEW_MIN_ANGLE_DEG or abs(angle) > DESKEW_MAX_ANGLE_DEG:
        return image

    logger.info("Detected slanted page (%.2f deg). Straightening before OCR.", angle)
    # `expand` keeps the corners of the rotated page inside the canvas, and the
    # white fill matches the scanned background.
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")


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
