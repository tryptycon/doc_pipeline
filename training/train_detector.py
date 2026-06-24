"""
Fine-tunes the pretrained tech4humans/yolov8s-signature-detector on YOUR
labeled documents (Indian thumbprint impressions, the specific routing-stamp
designs, your scan quality) -- starting from their checkpoint means a few
hundred labeled examples meaningfully improves precision, instead of needing
the tens of thousands of images a from-scratch YOLO would need.

Usage:
    python train_detector.py /path/to/dataset.yaml --epochs 100 --imgsz 1280

Notes:
- imgsz=1280 (not the YOLO default 640) matters here: a signature or a
  thumbprint is a small part of a full 300dpi scanned page, and the default
  640px training resolution will downsample it into a handful of pixels.
- If you've added a `stamp`/`thumbprint` class alongside `signature` in your
  Label Studio labels, this fine-tunes a multi-class detector in one model
  instead of needing a separate model per artifact type.
"""
import argparse
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def train(dataset_yaml: str, epochs: int, imgsz: int, batch: int, base_repo: str):
    base_weights = hf_hub_download(repo_id=base_repo, filename="yolov8s.pt")
    model = YOLO(base_weights)
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,           # early stop if val mAP plateaus
        project="runs/signature_finetune",
        name="exp",
    )
    print("Best weights saved under runs/signature_finetune/exp/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base-repo", default="tech4humans/yolov8s-signature-detector")
    args = parser.parse_args()
    train(args.dataset_yaml, args.epochs, args.imgsz, args.batch, args.base_repo)
