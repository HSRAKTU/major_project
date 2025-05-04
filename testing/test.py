import os
import csv
import argparse

def load_txt_ids(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def load_csv_ids(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return set(row["Design"] for row in reader)

def write_list(ids, out_path, label):
    with open(out_path, 'w') as f:
        for id_ in sorted(ids):
            f.write(id_ + '\n')
    print(f"✅ Saved {label} → {out_path} ({len(ids)} IDs)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/DrivAerNetPlusPlus_Drag_8k_cleaned.csv', help='CSV with Design column')
    parser.add_argument('--train', default='data/subset_dir/train_design_ids.txt')
    parser.add_argument('--val', default='data/subset_dir/val_design_ids.txt')
    parser.add_argument('--test', default='data/subset_dir/test_design_ids.txt')
    parser.add_argument('--out_dir', default='testing/missing_checks', help='Where to save mismatch reports')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_ids = load_csv_ids(args.csv)
    split_ids = load_txt_ids(args.train) | load_txt_ids(args.val) | load_txt_ids(args.test)

    only_in_csv = csv_ids - split_ids
    only_in_txts = split_ids - csv_ids

    print(f"\n📦 Total CSV Designs: {len(csv_ids)}")
    print(f"📦 Total .txt Split Designs: {len(split_ids)}")
    print(f"🔍 Only in CSV: {len(only_in_csv)}")
    print(f"🔍 Only in Splits (.txts): {len(only_in_txts)}")

    write_list(only_in_csv, os.path.join(args.out_dir, 'present_in_csv_but_not_txt.txt'), "Extra CSV IDs")
    write_list(only_in_txts, os.path.join(args.out_dir, 'present_in_txt_but_not_csv.txt'), "Missing in CSV IDs")

if __name__ == "__main__":
    main()
