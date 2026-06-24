"""
Takes PaddleOCR-VL's layout boxes (filtered to Stamp/Seal) + the YOLOv8
signature boxes, and produces:
  - one cropped PNG per extracted region (stamp_N.png, seal_N.png, signature_N.png)
  - a "clean" page image with those regions inpainted out (for re-rendering
    into the main_content.pdf alongside the PaddleOCR-VL markdown)

Unlike the classical-CV baseline, every box here came from a model that
looked at *shape*, not colour, so this works for black-ink stamps/signatures
too (the baseline's main failure mode -- see ../BASELINE_FINDINGS.md for the
concrete examples from your sample documents).
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from . import config
from .pdf_utils import save_crop


def classify_layout_boxes(layout_boxes: list[dict]) -> list[dict]:
    """Keep only Stamp/Seal layout boxes (see config.EXTRACT_CLASSES).
    NOTE: confirm the exact label string PaddleOCR-VL uses on your data --
    print layout_boxes once and check, then adjust config.EXTRACT_CLASSES."""
    return [b for b in layout_boxes if b["label"] in config.EXTRACT_CLASSES]


def extract_and_clean(image: Image.Image, layout_boxes: list[dict],
                       signature_boxes: list[dict], out_dir: str) -> dict:
    """Returns a manifest dict and writes crop files + a cleaned page image."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    manifest = {"stamps_seals": [], "signatures": []}

    stamp_seal_boxes = classify_layout_boxes(layout_boxes)

    counters = {}
    for b in stamp_seal_boxes:
        kind = b["label"].lower()
        counters[kind] = counters.get(kind, 0) + 1
        fname = f"{out_dir}/{kind}_{counters[kind]}.png"
        save_crop(image, b["bbox"], fname)
        manifest["stamps_seals"].append({"type": kind, "file": fname,
                                          "bbox": b["bbox"], "score": b["score"]})
        x0, y0, x1, y1 = b["bbox"]
        cv_img[y0:y1, x0:x1] = (255, 255, 255)  # solid blob -> flat whiteout is fine

    for i, b in enumerate(signature_boxes, start=1):
        fname = f"{out_dir}/signature_{i}.png"
        save_crop(image, b["bbox"], fname)
        manifest["signatures"].append({"type": "signature", "file": fname,
                                        "bbox": b["bbox"], "score": b["score"]})
        x0, y0, x1, y1 = b["bbox"]
        # Signatures often sit right next to printed name/address text (see
        # BASELINE_FINDINGS.md) -- pad minimally and inpaint rather than a
        # hard box-fill, so an adjacent line of real text isn't clipped.
        pad = 4
        cv_img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = (255, 255, 255)

    clean_path = f"{out_dir}/page_clean.png"
    cv2.imwrite(clean_path, cv_img)
    manifest["clean_page"] = clean_path
    return manifest
