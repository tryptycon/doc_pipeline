"""Writes the final per-document output bundle."""
import json
from pathlib import Path
from PIL import Image
from . import config
from .pdf_utils import images_to_pdf


def write_document_output(doc_id: str, page_results: list[dict]) -> dict:
    """
    page_results: one dict per page, each containing:
      - "clean_image": PIL.Image (main content, artifacts removed)
      - "markdown": str (transcribed main content for that page)
      - "manifest": dict from region_extractor.extract_and_clean (file paths/boxes)
    """
    out_dir = Path(config.OUTPUT_DIR) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_images = [p["clean_image"] for p in page_results]
    pdf_path = out_dir / "main_content.pdf"
    images_to_pdf(clean_images, str(pdf_path))

    md_path = out_dir / "main_content.md"
    md_path.write_text("\n\n---\n\n".join(p["markdown"] for p in page_results), encoding="utf-8")

    manifest = {
        "doc_id": doc_id,
        "num_pages": len(page_results),
        "main_content_pdf": str(pdf_path),
        "main_content_markdown": str(md_path),
        "pages": [p["manifest"] for p in page_results],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    return manifest
