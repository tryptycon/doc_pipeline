"""Orchestrates the full per-document pipeline: preprocess -> PaddleOCR-VL
(layout + transcription) -> signature detection -> region extraction ->
storage."""
import uuid
from pathlib import Path
from PIL import Image

from . import preprocessing, ocr_engine, signature_detector, region_extractor, storage, config
from .pdf_utils import load_image


def process_document(path: str, doc_id: str | None = None) -> dict:
    doc_id = doc_id or f"{Path(path).stem}_{uuid.uuid4().hex[:8]}"
    pages = load_image(path)

    page_results = []
    tmp_dir = Path(config.OUTPUT_DIR) / doc_id / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(pages, start=1):
        page = preprocessing.preprocess(page)
        page_path = tmp_dir / f"page_{i}.png"
        page.save(page_path)

        ocr_results = ocr_engine.run(str(page_path))
        ocr_result = ocr_results[0] if ocr_results else {"markdown": "", "layout_boxes": []}

        sig_boxes = signature_detector.detect(str(page_path))

        region_out_dir = Path(config.OUTPUT_DIR) / doc_id / f"page_{i}"
        manifest = region_extractor.extract_and_clean(
            page, ocr_result["layout_boxes"], sig_boxes, str(region_out_dir)
        )

        clean_image = Image.open(manifest["clean_page"])
        page_results.append({
            "clean_image": clean_image,
            "markdown": ocr_result["markdown"],
            "manifest": manifest,
        })

    return storage.write_document_output(doc_id, page_results)
