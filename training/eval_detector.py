"""
Evaluate a (fine-tuned or pretrained) detector on the held-out TEST split --
note this is `test`, not `val`: val is used during training for early
stopping, test is the number you should actually trust.

Usage:
    python eval_detector.py runs/signature_finetune/exp/weights/best.pt dataset.yaml
"""
import argparse
from ultralytics import YOLO


def evaluate(weights_path: str, dataset_yaml: str):
    model = YOLO(weights_path)
    metrics = model.val(data=dataset_yaml, split="test")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP@0.5:   {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
    print("\nPer-class mAP@0.5:")
    for i, name in metrics.names.items():
        print(f"  {name}: {metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path")
    parser.add_argument("dataset_yaml")
    args = parser.parse_args()
    evaluate(args.weights_path, args.dataset_yaml)
