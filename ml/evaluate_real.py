"""Evaluate the trained YOLO11n-cls model on the real test images.

Usage:
  python ml/evaluate_real.py [path/to/best.pt]

Reports top-1 accuracy, per-class accuracy, a confusion matrix, and the
fill-percentage error using the mapping:
  empty -> 0%, halffull -> 50%, full -> 100%
"""

import sys
from pathlib import Path

from ultralytics import YOLO

FILL = {"empty": 0, "halffull": 50, "full": 100}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ML_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ML_DIR / "models" / "v1" / "best.pt"
TEST_DIR = ML_DIR / "data" / "cls" / "test_real"


def main():
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    if not weights.exists():
        print(f"Weights not found: {weights}")
        sys.exit(1)

    model = YOLO(str(weights))
    names = list(FILL)

    rows = []  # (true_class, predicted_class, confidence)
    for cls in names:
        for img in sorted((TEST_DIR / cls).iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            result = model.predict(str(img), verbose=False)[0]
            pred = result.names[int(result.probs.top1)]
            rows.append((cls, pred, float(result.probs.top1conf)))

    if not rows:
        print(f"No test images found under {TEST_DIR}")
        sys.exit(1)

    correct = sum(t == p for t, p, _ in rows)
    print(f"Real test images: {len(rows)}")
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

    print("\nFill-% error (percentage points):")
    overall = sum(abs(FILL[p] - FILL[t]) for t, p, _ in rows) / len(rows)
    print(f"  Overall MAE: {overall:.1f}")
    for cls in names:
        sub = [r for r in rows if r[0] == cls]
        if sub:
            mae = sum(abs(FILL[p] - FILL[t]) for t, p, _ in sub) / len(sub)
            print(f"  {cls:<9} MAE: {mae:.1f}")

    wrong = [r for r in rows if r[0] != r[1]]
    if wrong:
        print(f"\nMisclassified: {len(wrong)}")
        for t, p, c in wrong:
            print(f"  true={t:<9} pred={p:<9} confidence={c:.2f}")


if __name__ == "__main__":
    main()
