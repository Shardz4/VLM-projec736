"""scripts/download_coco_val.py

Automated downloader for MS-COCO 2017 Validation split:
- 5,000 images (val2017.zip ~778 MB)
- 25,000 human captions (captions_val2017.json)

Downloads and extracts directly into data/coco/
"""

import os
from pathlib import Path
import urllib.request
import zipfile
import json
import sys


VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


class DownloadProgressBar:
    def __init__(self, filename: str):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = int(downloaded * 100 / total_size)
            if percent != self.last_percent and percent % 5 == 0:
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading {self.filename}: {percent}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
                self.last_percent = percent
        else:
            sys.stdout.write(f"\r  Downloading {self.filename}: {downloaded / (1024 * 1024):.1f} MB")
            sys.stdout.flush()


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"  [Exists] {dest_path.name} already downloaded.")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    progress_bar = DownloadProgressBar(dest_path.name)
    urllib.request.urlretrieve(url, dest_path, progress_bar)
    print(f"\n  [Done] Downloaded {dest_path.name}")


def main():
    print("=" * 70)
    print("MS-COCO 2017 VALIDATION RETRIEVAL DATASET DOWNLOADER")
    print("=" * 70)

    target_dir = Path("data/coco")
    target_dir.mkdir(parents=True, exist_ok=True)

    images_zip = target_dir / "val2017.zip"
    annotations_zip = target_dir / "annotations_trainval2017.zip"
    val2017_folder = target_dir / "val2017"
    captions_file = target_dir / "annotations" / "captions_val2017.json"

    # 1. Download images
    print("\n[1/4] Downloading val2017.zip (~778 MB)...")
    download_file(VAL_IMAGES_URL, images_zip)

    # 2. Extract images
    if val2017_folder.exists() and len(list(val2017_folder.glob("*.jpg"))) >= 5000:
        print("  [Exists] val2017 folder already contains 5,000 images.")
    else:
        print(f"\n[2/4] Extracting {images_zip.name} to {val2017_folder}...")
        with zipfile.ZipFile(images_zip, "r") as zf:
            zf.extractall(target_dir)
        print("  [Done] Extracted val2017 images.")

    # 3. Download annotations
    print("\n[3/4] Downloading annotations_trainval2017.zip (~241 MB)...")
    download_file(ANNOTATIONS_URL, annotations_zip)

    # 4. Extract captions_val2017.json only
    print("\n[4/4] Extracting captions_val2017.json...")
    (target_dir / "annotations").mkdir(parents=True, exist_ok=True)
    if captions_file.exists():
        print("  [Exists] captions_val2017.json already extracted.")
    else:
        with zipfile.ZipFile(annotations_zip, "r") as zf:
            # Extract only captions_val2017.json
            member = "annotations/captions_val2017.json"
            if member in zf.namelist():
                zf.extract(member, target_dir)
                print(f"  [Done] Extracted {captions_file}")
            else:
                print(f"  [Error] {member} not found in zip file!")

    # Verify counts
    n_images = len(list(val2017_folder.glob("*.jpg"))) if val2017_folder.exists() else 0
    n_captions = 0
    if captions_file.exists():
        with open(captions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            n_captions = len(data.get("annotations", []))

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    print(f"  Images Directory   : {val2017_folder.resolve()} ({n_images} images)")
    print(f"  Captions File      : {captions_file.resolve()} ({n_captions} captions)")
    print("=" * 70)

    if n_images >= 5000 and n_captions >= 25000:
        print("[SUCCESS] MS-COCO 2017 Validation dataset is 100% ready for retrieval evaluation!")
    else:
        print("[WARNING] Verification counts did not match expected (5,000 images, 25,000 captions).")


if __name__ == "__main__":
    main()
