"""Central config. Override any of these with environment variables."""
import os

# Layout detector (PP-DocLayoutV3) — categories we treat as "extract & remove"
# vs "keep as main content". Adjust EXTRACT_CLASSES if your PP-DocLayout
# version names them slightly differently (check `model.label_list`).
EXTRACT_CLASSES = {"Stamp", "Seal"}
MAIN_CONTENT_CLASSES = {"Text", "Title", "Paragraph Title", "Document Title",
                         "Table", "Figure Caption", "Header", "Footer",
                         "Number", "Abstract", "Content", "References"}

# Signature detector
SIGNATURE_MODEL_REPO = os.getenv("SIGNATURE_MODEL_REPO", "tech4humans/yolov8s-signature-detector")
SIGNATURE_CONF_THRESHOLD = float(os.getenv("SIGNATURE_CONF_THRESHOLD", "0.35"))

# OCR / transcription
OCR_MODEL_NAME = os.getenv("OCR_MODEL_NAME", "PaddlePaddle/PaddleOCR-VL")
OCR_LANGS = os.getenv("OCR_LANGS", "hi,en").split(",")
DEVICE = os.getenv("DEVICE", "cuda")  # falls back to "cpu" automatically if no GPU

# Storage
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
RENDER_DPI = int(os.getenv("RENDER_DPI", "300"))

# Batch / scaling
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
