import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def display_slices(
    slices,
    car_id=None,
    n_cols=5,
    limit=None,
    figsize=(15, 3),
    axis='x',
    save_path=None
):
    """
    Display or save 2D slices from a point cloud.
    slices: list of (Mi, 2) np arrays
    """
    if limit:
        slices = slices[:limit]

    all_points = np.vstack([sl for sl in slices if sl.size])
    xmin, xmax = all_points[:, 0].min(), all_points[:, 0].max()
    ymin, ymax = all_points[:, 1].min(), all_points[:, 1].max()

    pad_x = 0.02 * (xmax - xmin)
    pad_y = 0.02 * (ymax - ymin)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    n = len(slices)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))

    for idx, sl in enumerate(slices):
        ax = axes.flat[idx]
        if sl.size:
            ax.scatter(sl[:, 0], sl[:, 1], s=2, c='k')
        else:
            ax.text(0.5, 0.5, 'Empty', ha='center', va='center')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(f"{axis} ∈ slice {idx}", fontsize=8)

    for j in range(idx + 1, n_rows * n_cols):
        axes.flat[j].axis('off')

    fig.suptitle(f"Slices for {car_id}" if car_id else "Point Cloud Slices", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[✓] Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize saved 2D slices from point cloud")
    parser.add_argument('--input', required=True, help="Path to .npy file (saved slice list)")
    parser.add_argument('--cols', type=int, default=5, help="Number of columns in grid display")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of slices shown")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save output image (optional)")
    args = parser.parse_args()

    path = args.input
    assert os.path.exists(path), f"File not found: {path}"
    slices = np.load(path, allow_pickle=True)

    car_id = os.path.splitext(os.path.basename(path))[0]
    display_slices(
        slices,
        car_id=car_id,
        n_cols=args.cols,
        limit=args.limit,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
