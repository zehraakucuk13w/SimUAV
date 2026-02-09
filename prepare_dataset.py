"""
SimUAV Dataset Preparation Script
Prepares YOLO dataset structure with train/val/test splits.
Does NOT modify original dataset folder.
"""

import os
import shutil
import random
from pathlib import Path

# --- CONFIG ---
SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("yolo_dataset")
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05
RANDOM_SEED = 42

def create_directory_structure():
    """Create YOLO directory structure."""
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure created")

def get_image_list():
    """Get list of all images (without extension)."""
    images_dir = SOURCE_DIR / 'images'
    image_files = list(images_dir.glob('*.png'))
    basenames = [f.stem for f in image_files]
    print(f"✓ Found {len(basenames)} images")
    return basenames

def split_dataset(basenames):
    """Split dataset into train/val/test."""
    random.seed(RANDOM_SEED)
    random.shuffle(basenames)
    
    n = len(basenames)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    splits = {
        'train': basenames[:train_end],
        'val': basenames[train_end:val_end],
        'test': basenames[val_end:]
    }
    
    for split, items in splits.items():
        print(f"  {split}: {len(items)} images")
    
    return splits

def copy_files(splits):
    """Copy images and labels to their respective splits."""
    for split, basenames in splits.items():
        for basename in basenames:
            # Copy image
            src_img = SOURCE_DIR / 'images' / f'{basename}.png'
            dst_img = OUTPUT_DIR / 'images' / split / f'{basename}.png'
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_lbl = SOURCE_DIR / 'labels' / f'{basename}.txt'
            dst_lbl = OUTPUT_DIR / 'labels' / split / f'{basename}.txt'
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
        
        print(f"✓ Copied {split} files")

def create_data_yaml():
    """Create data.yaml configuration file."""
    yaml_content = """# SimUAV Dataset Configuration
# Air-to-Air UAV Detection

path: /content/yolo_dataset  # Colab path (update for local)
train: images/train
val: images/val
test: images/test

# Class names
names:
  0: enemy_uav

# Number of classes
nc: 1
"""
    
    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"✓ Created {yaml_path}")

def verify_dataset():
    """Verify dataset integrity."""
    print("\n--- Verification ---")
    for split in ['train', 'val', 'test']:
        img_count = len(list((OUTPUT_DIR / 'images' / split).glob('*.png')))
        lbl_count = len(list((OUTPUT_DIR / 'labels' / split).glob('*.txt')))
        status = "✓" if img_count == lbl_count else "✗"
        print(f"{status} {split}: {img_count} images, {lbl_count} labels")

def main():
    print("=" * 50)
    print("SimUAV Dataset Preparation")
    print("=" * 50)
    
    # Check source exists
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory '{SOURCE_DIR}' not found!")
        return
    
    create_directory_structure()
    basenames = get_image_list()
    splits = split_dataset(basenames)
    copy_files(splits)
    create_data_yaml()
    verify_dataset()
    
    print("\n" + "=" * 50)
    print("DONE! Next steps:")
    print("1. Zip yolo_dataset folder")
    print("2. Upload to Google Drive")
    print("3. Run train_yolo.ipynb in Colab")
    print("=" * 50)

if __name__ == "__main__":
    main()
