import os
import argparse
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SPLIT_TXT_PATHS = {
    "train": "data/subset_dir/train_design_ids.txt",
    "val": "data/subset_dir/val_design_ids.txt",
    "test": "data/subset_dir/test_design_ids.txt",
}

def pad_and_mask_slices(slice_list, target_slices=80, target_points=6500):
    padded = np.zeros((target_slices, target_points, 2), dtype=np.float32)
    point_mask = np.zeros((target_slices, target_points), dtype=np.float32)
    slice_mask = np.zeros((target_slices,), dtype=np.float32)

    for i, sl in enumerate(slice_list):
        if sl.shape[0] == 0:
            continue
        n_pts = min(sl.shape[0], target_points)
        padded[i, :n_pts] = sl[:n_pts]
        point_mask[i, :n_pts] = 1
        slice_mask[i] = 1

    return padded, point_mask, slice_mask


def load_design_ids(split):
    path = SPLIT_TXT_PATHS[split]
    with open(path, "r") as f:
        ids = [line.strip() for line in f.readlines()]
    return set(ids)


def process_all_slices(slice_dir, output_dir, split, target_slices=80, target_points=6500):
    os.makedirs(output_dir, exist_ok=True)
    design_ids = load_design_ids(split)
    logging.info(f"📂 Preparing split: {split} → {len(design_ids)} design IDs")

    all_files = [f for f in os.listdir(slice_dir) if f.endswith(".npy")]
    matched_files = [f for f in all_files if any(f.startswith(design_id + "_") for design_id in design_ids)]

    logging.info(f"🎯 Matched {len(matched_files)} files for split '{split}' in {slice_dir}")

    for f in tqdm(matched_files, desc=f"Padding/masking {split}", ncols=80):
        path = os.path.join(slice_dir, f)
        try:
            slices = np.load(path, allow_pickle=True)
            padded, point_mask, slice_mask = pad_and_mask_slices(slices, target_slices, target_points)
            car_id = f.replace(".npy", "")
            out_path = os.path.join(output_dir, f"{car_id}.npz")
            np.savez_compressed(out_path, slices=padded, point_mask=point_mask, slice_mask=slice_mask)
        except Exception as e:
            logging.warning(f"❌ Failed: {f} → {str(e)}")

    logging.info(f"✅ Done. Processed {len(matched_files)} files into {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pad/mask 2D slices based on split")
    parser.add_argument('--split', type=str, choices=["train", "val", "test"], required=True,
                        help="Which split to process (train/val/test)")
    parser.add_argument('--slice_dir', type=str, default="outputs/slices", help="Dir with .npy slice files")
    parser.add_argument('--output_dir', type=str, default="outputs/pad_masked_slices", help="Output dir for .npz")
    parser.add_argument('--target_slices', type=int, default=80)
    parser.add_argument('--target_points', type=int, default=6500)
    args = parser.parse_args()

    process_all_slices(
        slice_dir=args.slice_dir,
        output_dir=args.output_dir,
        split=args.split,
        target_slices=args.target_slices,
        target_points=args.target_points
    )


if __name__ == "__main__":
    main()
