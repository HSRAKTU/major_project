import pandas as pd
import os

# --- Step 1: Load CSV and .txt files for checking ---
CSV_PATH = "data/DrivAerNetPlusPlus_Drag_8k.csv"
TXT_DIR = "data/subset_dir"

# Read original CSV
df = pd.read_csv(CSV_PATH)

# Combine all split IDs from train, val, test
split_ids = set()
for fname in ["train_design_ids.txt", "val_design_ids.txt", "test_design_ids.txt"]:
    path = os.path.join(TXT_DIR, fname)
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        split_ids.update(lines)

print(f"✅ Loaded CSV rows: {len(df)}")
print(f"✅ Loaded .txt-based split IDs: {len(split_ids)}")

# --- Step 2: Define helper to pad final segment of design ID ---
def pad_design_id(design_id):
    parts = design_id.split('_')
    if parts[-1].isdigit():
        parts[-1] = parts[-1].zfill(3)  # Zero pad to 3 digits
    return '_'.join(parts)

# --- Step 3: Apply padding only if needed and log changes ---
modified_count = 0
original_ids = df["Design"].tolist()
fixed_ids = []

for design in original_ids:
    padded = pad_design_id(design)
    if padded in split_ids:
        if design != padded:
            modified_count += 1
        fixed_ids.append(padded)
    else:
        # Keep original if padded version doesn't exist in any split
        fixed_ids.append(design)

df["Design"] = fixed_ids

print(f"\n🔧 Modified design IDs: {modified_count}")
print("💾 Saving cleaned CSV to: data/DrivAerNetPlusPlus_Drag_8k_cleaned.csv")

# --- Step 4: Save new cleaned version ---
df.to_csv("data/DrivAerNetPlusPlus_Drag_8k_cleaned.csv", index=False)
