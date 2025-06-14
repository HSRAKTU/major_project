import os
import argparse
import numpy as np

def read_design_ids(split_file):
    with open(split_file, "r") as f:
        return sorted([line.strip() for line in f if line.strip()])

def extract_ids_from_folder(folder, suffix):
    ids = []
    for fname in os.listdir(folder):
        if fname.endswith(suffix):
            base = fname.replace(suffix, "")
            ids.append(base)
    return sorted(ids)

def validate(split_name, split_path, slice_dir, padded_dir):
    print(f"\n🔎 Validating split: {split_name.upper()}")
    design_ids = read_design_ids(split_path)
    expected_count = len(design_ids)
    print(f"📦 Expected from {split_name}_design_ids.txt: {expected_count}")

    # Construct actual filenames expected
    slice_suffix = "_axis-x.npy"
    padded_suffix = "_axis-x.npz"

    sliced_ids = extract_ids_from_folder(slice_dir, slice_suffix)
    padded_ids = extract_ids_from_folder(padded_dir, padded_suffix)

    # Add suffix to design_ids for comparison
    expected_sliced = sorted([f"{d}{slice_suffix}" for d in design_ids])
    expected_padded = sorted([f"{d}{padded_suffix}" for d in design_ids])

    actual_sliced = sorted([f for f in os.listdir(slice_dir) if f.endswith(slice_suffix)])
    actual_padded = sorted([f for f in os.listdir(padded_dir) if f.endswith(padded_suffix)])

    # Compare
    missing_sliced = sorted(set(expected_sliced) - set(actual_sliced))
    missing_padded = sorted(set(expected_padded) - set(actual_padded))

    print(f"✅ Sliced found: {len(actual_sliced)} / {expected_count}")
    print(f"✅ Padded+Masked found: {len(actual_padded)} / {expected_count}")

    if missing_sliced:
        print(f"❌ Missing sliced files ({len(missing_sliced)}):")
        print(missing_sliced[:10], "..." if len(missing_sliced) > 10 else "")
    if missing_padded:
        print(f"❌ Missing padded files ({len(missing_padded)}):")
        print(missing_padded[:10], "..." if len(missing_padded) > 10 else "")
    if not missing_sliced and not missing_padded:
        print("🎉 All files present and matched.\n")

def main():
    parser = argparse.ArgumentParser(description="Validate slicing and padding pipeline")
    parser.add_argument('--subset_dir', default="data/subset_dir", help="Folder with split .txt files")
    parser.add_argument('--slice_dir', default="outputs/slices", help="Folder with sliced .npy files")
    parser.add_argument('--padded_dir', default="outputs/pad_masked_slices", help="Folder with padded .npz files")
    args = parser.parse_args()

    validate("train", os.path.join(args.subset_dir, "train_design_ids.txt"), args.slice_dir, args.padded_dir)
    validate("val", os.path.join(args.subset_dir, "val_design_ids.txt"), args.slice_dir, args.padded_dir)
    validate("test", os.path.join(args.subset_dir, "test_design_ids.txt"), args.slice_dir, args.padded_dir)

if __name__ == "__main__":
    main()
