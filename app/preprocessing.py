"""Deskew + denoise a page image before layout detection / OCR.

This stage is deliberately conservative: small, robust corrections only.
Aggressive thresholding/binarisation is left out on purpose -- PaddleOCR-VL
and PP-DocLayout are both trained on natural greyscale/colour scans and do
better with that than with a hard-binarised image.
"""
import cv2
import numpy as np
from PIL import Image


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def deskew(img: Image.Image) -> Image.Image:
    cv_img = _pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return img
    h, w = cv_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return _cv_to_pil(rotated)


def denoise(img: Image.Image) -> Image.Image:
    cv_img = _pil_to_cv(img)
    den = cv2.fastNlMeansDenoisingColored(cv_img, None, 5, 5, 7, 21)
    return _cv_to_pil(den)


def preprocess(img: Image.Image) -> Image.Image:
    return denoise(deskew(img))
