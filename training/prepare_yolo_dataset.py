"""
Takes a Label Studio export (YOLO format: images/ + labels/ + classes.txt)
and produces a train/val/test split ready for Ultralytics training.

Label Studio export structure expected:
    export/
      images/*.jpg
      labels/*.txt        (one .txt per image, YOLO box format)
      classes.txt         (one class name per line, in cls_id order)

Usage:
    python prepare_yolo_dataset.py /path/to/export /path/to/dataset_out \
        --val-frac 0.15 --test-frac 0.15
"""
import argparse
import random
import shutil
from pathlib import Path


def split_dataset(export_dir: Path, out_dir: Path, val_frac: float, test_frac: float, seed: int = 42):
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    classes_file = export_dir / "classes.txt"

    image_paths = sorted([p for p in images_dir.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    random.Random(seed).shuffle(image_paths)

    n = len(image_paths)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    splits = {
        "test": image_paths[:n_test],
        "val": image_paths[n_test:n_test + n_val],
        "train": image_paths[n_test + n_val:],
    }

    for split, paths in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, out_dir / split / "images" / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, out_dir / split / "labels" / label_path.name)
            else:
                # image with no boxes -> empty label file (valid in YOLO format)
                (out_dir / split / "labels" / (img_path.stem + ".txt")).touch()
        print(f"{split}: {len(paths)} images")

    classes = classes_file.read_text(encoding="utf-8").strip().splitlines()
    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(classes)) + "\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()
    split_dataset(args.export_dir, args.out_dir, args.val_frac, args.test_frac)
