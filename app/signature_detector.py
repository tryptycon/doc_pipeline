"""
Signature detection via tech4humans/yolov8s-signature-detector (Ultralytics
YOLOv8s, fine-tuned on the Tobacco800 + signatures-xc8up datasets).
Verified API against the model's Hugging Face README (94.74% precision /
89.72% recall / mAP@0.5 94.50% on its own test set) -- not executed in this
sandbox (no huggingface.co access here), so smoke-test on your own machine
before trusting it in production.

Licensing note: these YOLOv8s weights are AGPL-3.0 (inherited from
Ultralytics YOLOv8) -- fine for internal/offline batch processing, but check
your obligations if this is embedded in a service you ship externally. If
that's a blocker, tech4humans also publishes a Conditional-DETR-ResNet50
signature detector under Apache-2.0 with similar accuracy (93.65% mAP@0.5,
slightly slower) -- see `signature_detector_detr.py` stub below for the swap.
"""
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from . import config

_model = None


def get_model():
    global _model
    if _model is None:
        model_path = hf_hub_download(
            repo_id=config.SIGNATURE_MODEL_REPO,
            filename="yolov8s.pt",
        )
        _model = YOLO(model_path)
    return _model


def detect(image_path: str, conf: float = config.SIGNATURE_CONF_THRESHOLD) -> list[dict]:
    """Returns [{"bbox": (x0,y0,x1,y1), "score": float}, ...] for one image."""
    model = get_model()
    results = model(image_path, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            boxes.append({
                "bbox": (int(x0), int(y0), int(x1), int(y1)),
                "score": float(box.conf[0]),
            })
    return boxes


# --- Apache-2.0 alternative (use if AGPL is a problem for your deployment) ---
# from transformers import AutoImageProcessor, AutoModelForObjectDetection
# import torch
# from PIL import Image
#
# _detr_model_name = "tech4humans/conditional-detr-50-signature-detector"
# _processor = AutoImageProcessor.from_pretrained(_detr_model_name)
# _detr_model = AutoModelForObjectDetection.from_pretrained(_detr_model_name)
#
# def detect_detr(image_path: str, threshold: float = 0.5) -> list[dict]:
#     image = Image.open(image_path).convert("RGB")
#     inputs = _processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = _detr_model(**inputs)
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = _processor.post_process_object_detection(
#         outputs, target_sizes=target_sizes, threshold=threshold)[0]
#     return [{"bbox": tuple(round(v, 1) for v in box.tolist()), "score": float(score)}
#             for score, box in zip(results["scores"], results["boxes"])]
