import os
import shutil
import pandas as pd

"""
Creating train and validation sets from the EuroSAT dataset csv files
"""

source_root = 'EuroSAT'
val_csv = 'EuroSAT/val.csv'
dest_root = 'train'

# Load validation image relative paths from CSV
val_df = pd.read_csv(val_csv, header=None, dtype=str, index_col=False, skiprows=[0])
val_image_paths = val_df[1].tolist()

# Create destination root directory
os.makedirs(dest_root, exist_ok=True)

# Copy each image
for rel_path in val_image_paths:
    src_path = os.path.join(source_root, rel_path)
    dst_path = os.path.join(dest_root, rel_path)

    # Make sure the destination subfolder exists
    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)

    # Copy file
    shutil.copy2(src_path, dst_path)

print(f"Validation images copied to: {dest_root}")

