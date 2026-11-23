"""Train a YOLO11 classification model on ml/data/cls.

Usage (from the project root):
  python ml/train.py --name v2

Defaults are the v2 settings:
  yolo11s-cls, imgsz 320, stronger brightness/rotation/scale augmentation.
Pass other values to change them, e.g. --model yolo11n-cls.pt --imgsz 224 to
reproduce v1.

Weights are saved to ml/runs/<name>/weights/best.pt. Pretrained starting
weights are downloaded into ml/pretrained/ so they don't land in the project root.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data" / "cls"
RUNS_DIR = ML_DIR / "runs"
PRETRAINED_DIR = ML_DIR / "pretrained"


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO11 classification model.")
    parser.add_argument("--name", required=True, help="run name, e.g. v2")
    parser.add_argument("--model", default="yolo11s-cls.pt", help="pretrained model to start from")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hsv-v", type=float, default=0.6, help="brightness augmentation (v1 default 0.4)")
    parser.add_argument("--degrees", type=float, default=15.0, help="rotation augmentation (v1 default 0)")
    parser.add_argument("--scale", type=float, default=0.5, help="scale augmentation (v1 default 0.5)")
    parser.add_argument("--erasing", type=float, default=0.4, help="random erasing probability")
    args = parser.parse_args()

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(PRETRAINED_DIR / args.model))

    model.train(
        data=str(DATA_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        seed=args.seed,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        scale=args.scale,
        erasing=args.erasing,
        project=str(RUNS_DIR),
        name=args.name,
    )

    best = RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print(f"Evaluate with: python ml/evaluate_real.py {best}")


if __name__ == "__main__":
    main()
