"""PDF <-> image conversion and PDF (re)assembly."""
import fitz  # PyMuPDF
from PIL import Image
import io
from pathlib import Path
from . import config


def pdf_to_images(pdf_path: str, dpi: int = config.RENDER_DPI) -> list[Image.Image]:
    """Render every page of a PDF to a PIL Image at the given DPI."""
    images = []
    pdf = fitz.open(pdf_path)
    for page in pdf:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img.convert("RGB"))
    pdf.close()
    return images


def load_image(path: str) -> list[Image.Image]:
    """Accept a plain image file too (single 'page')."""
    if path.lower().endswith(".pdf"):
        return pdf_to_images(path)
    return [Image.open(path).convert("RGB")]


def images_to_pdf(images: list[Image.Image], out_path: str):
    """Assemble a list of (cleaned) page images into a single PDF."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not images:
        raise ValueError("No images to assemble into a PDF")
    first, rest = images[0], images[1:]
    first.save(out_path, save_all=True, append_images=rest)


def save_crop(image: Image.Image, box: tuple[int, int, int, int], out_path: str, pad: int = 12):
    """Crop a region (with a little padding) and save it as its own image file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = image.size
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    image.crop((x0, y0, x1, y1)).save(out_path)
