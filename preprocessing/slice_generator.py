import os
import glob
import logging
import argparse
import numpy as np
import paddle
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_design_ids(split, subset_dir="data/subset_dir"):
    split_file = os.path.join(subset_dir, f"{split}_design_ids.txt")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        ids = {line.strip() for line in f if line.strip()}
    logging.info(f"Loaded {len(ids)} design IDs for split: {split}")
    return ids


class PointCloudSlicer:
    def __init__(self,
                 input_dir="data/PointClouds",
                 output_dir="outputs/slices",
                 num_slices=80,
                 axis='x',
                 max_files=None,
                 split="train",
                 subset_dir="data/subset_dir"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_slices = num_slices
        self.axis = axis
        self.max_files = max_files
        self.split = split
        self.subset_dir = subset_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        if self.axis not in self.axis_map:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

        self.valid_ids = load_design_ids(split, subset_dir)

    def load_point_cloud(self, file_path):
        tensor = paddle.load(file_path)
        return tensor.numpy()

    def generate_slices(self, points):
        ax = self.axis_map[self.axis]
        coords = points[:, ax]
        min_val, max_val = coords.min(), coords.max()
        edges = np.linspace(min_val, max_val, self.num_slices + 1)

        slices = []
        for i in range(self.num_slices):
            low, high = edges[i], edges[i + 1]
            mask = (coords >= low) & (coords < high)
            sl = points[mask]
            sl = np.delete(sl, ax, axis=1)
            slices.append(sl)
        return slices

    def process_file(self, file_path):
        points = self.load_point_cloud(file_path)
        slices = self.generate_slices(points)
        total_points = sum(len(sl) for sl in slices)
        logging.info(f"{os.path.basename(file_path)} → {total_points} total points across {len(slices)} slices.")
        return slices

    def run(self):
        all_files = sorted(glob.glob(os.path.join(self.input_dir, "*.paddle_tensor")))
        filtered_files = [
            f for f in all_files
            if os.path.splitext(os.path.basename(f))[0] in self.valid_ids
        ]

        if self.max_files is not None:
            filtered_files = filtered_files[:self.max_files]

        logging.info(f"Processing {len(filtered_files)} point clouds from split: {self.split}")

        count_success, count_error = 0, 0

        for file_path in tqdm(filtered_files, desc=f"Slicing {self.split} set"):
            try:
                car_id = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(
                    self.output_dir, f"{car_id}_axis-{self.axis}.npy"
                )
                slices = self.process_file(file_path)
                np.save(output_path, np.array(slices, dtype=object), allow_pickle=True)
                logging.info(f"Sliced and saved: {car_id} → {output_path}")
                count_success += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                count_error += 1

        logging.info(f"Finished slicing split '{self.split}': {count_success} succeeded, {count_error} failed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Slice 3D point clouds into 2D slices using predefined train/val/test split")
    parser.add_argument('--input_dir', type=str, default="data/PointClouds", help="Directory with .paddle_tensor files")
    parser.add_argument('--output_dir', type=str, default="outputs/slices", help="Where to save .npy slices")
    parser.add_argument('--num_slices', type=int, default=80, help="Number of slices along axis")
    parser.add_argument('--axis', type=str, default='x', choices=['x', 'y', 'z'], help="Axis to slice along")
    parser.add_argument('--max_files', type=int, default=None, help="Max files to slice (None = all)")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help="Dataset split to process")
    parser.add_argument('--subset_dir', type=str, default='data/subset_dir', help="Directory containing split ID .txt files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    slicer = PointCloudSlicer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_slices=args.num_slices,
        axis=args.axis,
        max_files=args.max_files,
        split=args.split,
        subset_dir=args.subset_dir
    )
    slicer.run()
