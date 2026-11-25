"""Evaluate a trained YOLO11 classification model on a labelled test folder.

Usage (from the project root):
  python ml/evaluate_real.py [path/to/best.pt] [--test path/to/test_folder]

The test folder has one sub-folder per class (empty, halffull, full, and
optionally no_bin). Defaults: the approved v1 weights and ml/data/cls/test_real.

Reports top-1 accuracy, per-class accuracy, a confusion matrix, and the
fill-percentage error using the mapping:
  empty -> 0%, halffull -> 50%, full -> 100%
no_bin has no fill percentage, so it is left out of the fill-% error.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

FILL = {"empty": 0, "halffull": 50, "full": 100, "no_bin": None}
CLASS_ORDER = ["empty", "halffull", "full", "no_bin"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ML_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ML_DIR / "models" / "v1" / "best.pt"
DEFAULT_TEST = ML_DIR / "data" / "cls" / "test_real"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO11 classification model.")
    parser.add_argument("weights", nargs="?", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--test", default=str(DEFAULT_TEST), help="test folder with one sub-folder per class")
    args = parser.parse_args()

    weights = Path(args.weights)
    test_dir = Path(args.test)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not test_dir.exists():
        raise SystemExit(f"Test folder not found: {test_dir}")

    model = YOLO(str(weights))

    rows = []  # (true_class, predicted_class, confidence)
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        true_cls = cls_dir.name
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            result = model.predict(str(img), verbose=False)[0]
            pred = result.names[int(result.probs.top1)]
            rows.append((true_cls, pred, float(result.probs.top1conf)))

    if not rows:
        raise SystemExit(f"No test images found under {test_dir}")

    seen = {r[0] for r in rows} | {r[1] for r in rows}
    names = [c for c in CLASS_ORDER if c in seen] + sorted(seen - set(CLASS_ORDER))

    correct = sum(t == p for t, p, _ in rows)
    print(f"Test folder:      {test_dir}")
    print(f"Test images:      {len(rows)}")
    print(f"Top-1 accuracy:   {correct / len(rows):.1%}\n")

    print("Per-class accuracy:")
    for cls in names:
        sub = [r for r in rows if r[0] == cls]
        if sub:
            acc = sum(t == p for t, p, _ in sub) / len(sub)
            print(f"  {cls:<9} n={len(sub):<4} acc={acc:.1%}")

    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(" " * 10 + "".join(f"{n:>10}" for n in names))
    for t in names:
        counts = "".join(f"{sum(1 for a, p, _ in rows if a == t and p == q):>10}" for q in names)
        print(f"{t:<10}{counts}")

    print("\nFill-% error (percentage points), using images where both values have a fill %:")
    pairs = [(t, p) for t, p, _ in rows if FILL.get(t) is not None and FILL.get(p) is not None]
    if pairs:
        overall = sum(abs(FILL[p] - FILL[t]) for t, p in pairs) / len(pairs)
        print(f"  Overall MAE: {overall:.1f}  (n={len(pairs)})")
        for cls in ["empty", "halffull", "full"]:
            sub = [(t, p) for t, p in pairs if t == cls]
            if sub:
                mae = sum(abs(FILL[p] - FILL[t]) for t, p in sub) / len(sub)
                print(f"  {cls:<9} MAE: {mae:.1f}")
    else:
        print("  No images with a fill % on both sides.")

    wrong = [r for r in rows if r[0] != r[1]]
    if wrong:
        print(f"\nMisclassified: {len(wrong)}")
        for t, p, c in wrong:
            print(f"  true={t:<9} pred={p:<9} confidence={c:.2f}")


if __name__ == "__main__":
    main()
