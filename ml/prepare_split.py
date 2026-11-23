"""Prepare the waste-bin dataset for YOLO11 classification.

Copies images from ml/data/raw into ml/data/cls:
  cls/train/<class>/     80% of simulated images (seeded shuffle)
  cls/val/<class>/       20% of simulated images
  cls/test_real/<class>/ all real images (never trained on)

Classes: empty, halffull, full
"""

import random
import shutil
from pathlib import Path

SEED = 0
TRAIN_RATIO = 0.8
CLASSES = ["empty", "halffull", "full"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ML_DIR = Path(__file__).resolve().parent
RAW = ML_DIR / "data" / "raw"
OUT = ML_DIR / "data" / "cls"


def images_in(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def copy_files(files, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)

    rng = random.Random(SEED)
    summary = {}

    for cls in CLASSES:
        sim = images_in(RAW / "simulated" / cls)
        rng.shuffle(sim)
        cut = int(len(sim) * TRAIN_RATIO)
        train, val = sim[:cut], sim[cut:]
        copy_files(train, OUT / "train" / cls)
        copy_files(val, OUT / "val" / cls)

        real = images_in(RAW / "real" / cls)
        copy_files(real, OUT / "test_real" / cls)

        summary[cls] = (len(train), len(val), len(real))

    print(f"{'class':<10}{'train':>8}{'val':>8}{'test_real':>12}")
    for cls, (t, v, r) in summary.items():
        print(f"{cls:<10}{t:>8}{v:>8}{r:>12}")
    print(f"\nWritten to: {OUT}")


if __name__ == "__main__":
    main()
