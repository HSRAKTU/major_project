import os
import pandas as pd

# Paths
CSV_PATH = "data/DrivAerNetPlusPlus_Drag_8k.csv"
TRAIN_PATH = "data/subset_dir/train_design_ids.txt"
VAL_PATH = "data/subset_dir/val_design_ids.txt"
TEST_PATH = "data/subset_dir/test_design_ids.txt"

# Load design IDs from CSV
df = pd.read_csv(CSV_PATH)
csv_ids = set(df["Design"].astype(str))

# Load design IDs from each split
def load_ids(txt_path):
    with open(txt_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

train_ids = load_ids(TRAIN_PATH)
val_ids = load_ids(VAL_PATH)
test_ids = load_ids(TEST_PATH)

# Merge all known IDs
all_split_ids = train_ids | val_ids | test_ids

# Compare
unmatched_csv_ids = csv_ids - all_split_ids
matched_csv_ids = csv_ids & all_split_ids

# Report
print(f"🔎 Total entries in Cd CSV: {len(csv_ids)}")
print(f"✅ Found in train/val/test: {len(matched_csv_ids)}")
print(f"❌ Not found in any split: {len(unmatched_csv_ids)}")

if unmatched_csv_ids:
    print("\n🔍 Example missing IDs:", list(unmatched_csv_ids)[:10])
