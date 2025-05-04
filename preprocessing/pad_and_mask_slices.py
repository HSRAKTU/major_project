import os
import argparse
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def pad_and_mask_slices(slice_list, target_slices=80, target_points=6500):
    padded = np.zeros((target_slices, target_points, 2), dtype=np.float32)
    point_mask = np.zeros((target_slices, target_points), dtype=np.float32)
    slice_mask = np.zeros((target_slices,), dtype=np.float32)

    for i, sl in enumerate(slice_list):
        if sl.shape[0] == 0:
            continue  # empty slice, keep zeroed
        n_pts = min(sl.shape[0], target_points)
        padded[i, :n_pts] = sl[:n_pts]
        point_mask[i, :n_pts] = 1
        slice_mask[i] = 1

    return padded, point_mask, slice_mask


def process_all_slices(slice_dir, output_dir, target_slices=80, target_points=6500):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(slice_dir) if f.endswith(".npy")]

    logging.info(f"Found {len(files)} sliced cars in {slice_dir}. Starting padding...")

    for f in tqdm(files, desc="Padding and masking", ncols=80):
        path = os.path.join(slice_dir, f)
        try:
            slices = np.load(path, allow_pickle=True)
            padded, point_mask, slice_mask = pad_and_mask_slices(slices, target_slices, target_points)

            car_id = f.replace(".npy", "")
            out_path = os.path.join(output_dir, f"{car_id}.npz")
            np.savez_compressed(out_path, slices=padded, point_mask=point_mask, slice_mask=slice_mask)
        except Exception as e:
            logging.warning(f"❌ Failed: {f} → {str(e)}")

    logging.info(f"✅ Done. Padded and masked {len(files)} files to {output_dir}")



def main():
    parser = argparse.ArgumentParser(description="Pad and mask 2D point cloud slices for training")
    parser.add_argument('--slice_dir', type=str, default="outputs/slices", help="Directory with raw slice .npy files")
    parser.add_argument('--output_dir', type=str, default="outputs/pad_masked_slices", help="Where to save padded .npz files")
    parser.add_argument('--target_slices', type=int, default=80, help="Number of slices per car")
    parser.add_argument('--target_points', type=int, default=6500, help="Points per slice (after padding)")
    args = parser.parse_args()

    process_all_slices(
        slice_dir=args.slice_dir,
        output_dir=args.output_dir,
        target_slices=args.target_slices,
        target_points=args.target_points
    )


if __name__ == "__main__":
    main()
