"""
=============================================================
  Road Classifier - Data Preparation Script

  This script helps you organise your dataset into the
  correct folder structure required for training.

  Expected final structure:
    data/
      train/
        road/        ← road images for training
        not_road/    ← non-road images for training
      val/
        road/        ← road images for validation
        not_road/    ← non-road images for validation
      test/          ← optional: unlabelled images to test

  Usage:
    python prepare_data.py --source your_images/ --split 0.8
=============================================================
"""

import os
import shutil
import random
import argparse
from pathlib import Path


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder):
    return sum(
        1 for f in Path(folder).rglob("*")
        if f.suffix.lower() in SUPPORTED
    )


def split_and_copy(source_dir, dest_dir, label, split_ratio=0.8):
    """
    Copies images from source_dir into dest_dir/train/<label>
    and dest_dir/val/<label> according to split_ratio.
    """
    images = [
        f for f in Path(source_dir).rglob("*")
        if f.suffix.lower() in SUPPORTED
    ]
    random.shuffle(images)

    split_idx  = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    for phase, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dest = Path(dest_dir) / phase / label
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dest / img.name)

    print(f"  [{label}]  Train: {len(train_imgs)}  Val: {len(val_imgs)}")
    return len(train_imgs), len(val_imgs)


def verify_structure(data_dir="data"):
    """Check if data folder structure is correct."""
    required = [
        "data/train/road",
        "data/train/not_road",
        "data/val/road",
        "data/val/not_road",
    ]

    print("\n─── Verifying Dataset Structure ───")
    ok = True
    for folder in required:
        path = Path(folder)
        if path.exists():
            count = count_images(str(path))
            status = "✅" if count > 0 else "⚠️  (empty!)"
            print(f"  {status}  {folder}  ({count} images)")
            if count == 0:
                ok = False
        else:
            print(f"  ❌  {folder}  (MISSING)")
            ok = False

    if ok:
        print("\n✅ Dataset structure looks good! You can now run train.py")
    else:
        print("\n⚠️  Fix the above issues before training.")
    return ok


def show_recommended_sources():
    print("""
╔══════════════════════════════════════════════════════╗
║         Recommended Free Datasets for Road           ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  1. KITTI Dataset (road images)                      ║
║     http://www.cvlibs.net/datasets/kitti/            ║
║                                                      ║
║  2. Mapillary Vistas (street-level images)           ║
║     https://www.mapillary.com/dataset/vistas         ║
║                                                      ║
║  3. Kaggle: Road vs Non-Road                         ║
║     https://www.kaggle.com/datasets                  ║
║     (search: "road classification dataset")         ║
║                                                      ║
║  4. Google Images / Your own phone photos            ║
║     - Take ~200 road photos yourself                 ║
║     - Take ~200 non-road photos yourself             ║
║     → 400 images is enough to start!                 ║
║                                                      ║
║  MINIMUM RECOMMENDED:                                ║
║    • 300-500 road images                             ║
║    • 300-500 not_road images                         ║
║    (More = better accuracy)                          ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Prepare road dataset")
    parser.add_argument("--road",    type=str, help="Folder of road images")
    parser.add_argument("--notroad", type=str, help="Folder of non-road images")
    parser.add_argument("--split",   type=float, default=0.8,
                        help="Train/Val split ratio (default: 0.8)")
    parser.add_argument("--verify",  action="store_true",
                        help="Just verify existing structure")
    parser.add_argument("--sources", action="store_true",
                        help="Show recommended dataset sources")
    args = parser.parse_args()

    if args.sources:
        show_recommended_sources()
        return

    if args.verify:
        verify_structure()
        return

    if not args.road or not args.notroad:
        print("Usage examples:")
        print("  python prepare_data.py --road my_road_pics/ --notroad my_other_pics/")
        print("  python prepare_data.py --verify")
        print("  python prepare_data.py --sources")
        return

    print(f"\nPreparing dataset (split: {args.split*100:.0f}% train / {(1-args.split)*100:.0f}% val)")
    print("─" * 45)

    split_and_copy(args.road,    "data", "road",     args.split)
    split_and_copy(args.notroad, "data", "not_road", args.split)

    verify_structure()


if __name__ == "__main__":
    main()
