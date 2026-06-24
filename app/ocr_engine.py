"""
Wraps PaddleOCR-VL (https://www.paddleocr.ai/.../PaddleOCR-VL.html), which is
ALREADY a combined layout-detection + VLM-transcription pipeline -- you do not
need to run PP-DocLayout separately, predict() returns both the layout boxes
(res['layout_det_res']['boxes'], each with a 'label'/'cls_id'/'coordinate')
and the transcribed content (res.save_to_markdown / res['markdown']).

Verified against the official usage tutorial (accurate as of the PaddleOCR-VL
v1.6 docs, May 2026). NOT executed in this sandbox -- this environment has no
route to huggingface.co or Baidu's model CDN to download weights, so test this
against a real page on your own GPU/CPU machine before trusting it in
production. The one thing to double check on your own data: the exact layout
label string PaddleOCR-VL uses for stamps/seals (we filter on
{"seal", "stamp"} below as the most likely names given the docs' explicit
"seal recognition" feature -- print res['layout_det_res']['boxes'] once on a
sample page and adjust config.EXTRACT_CLASSES if the actual label differs).
"""
from paddleocr import PaddleOCRVL
from . import config

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = PaddleOCRVL(
            use_seal_recognition=True,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,  # these scans are flat, not photographed
            device=config.DEVICE,
            # Keep stamp/seal text out of the main-content markdown -- we
            # transcribe those separately if useful, but the user wants the
            # body letter, not the routing-stamp text, in main_content.md
            markdown_ignore_labels=["number", "footnote", "header", "header_image",
                                     "footer", "footer_image", "aside_text",
                                     "seal", "stamp"],
        )
    return _pipeline


def run(image_path: str) -> list[dict]:
    """Runs the full PaddleOCR-VL pipeline on one image/PDF page.

    Returns a list (one entry per page, but we always feed single pages)
    of dicts: {"markdown": str, "layout_boxes": [{"label","score","bbox"}]}
    """
    pipeline = get_pipeline()
    output = pipeline.predict(image_path)
    results = []
    for res in output:
        layout_boxes = []
        for box in res.get("layout_det_res", {}).get("boxes", []):
            x0, y0, x1, y1 = box["coordinate"]
            layout_boxes.append({
                "label": box["label"],
                "score": float(box["score"]),
                "bbox": (int(x0), int(y0), int(x1), int(y1)),
            })
        results.append({
            "markdown": res.get("markdown", {}).get("markdown_texts", "")
                        if isinstance(res.get("markdown"), dict) else str(res.get("markdown", "")),
            "layout_boxes": layout_boxes,
            "raw": res,  # kept so storage.py can still call res.save_to_json/markdown directly
        })
    return results
