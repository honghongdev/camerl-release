import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
# CA dataset directory produced by generate.py (NPZ_SAVE_FOLDER)
CA_FOLDER = "../saved/2-dataset/normal_D5-50_r3_ca/"

# File index to view (0-based, maps to collision_images_000.npz, 001.npz, ...)
DEFAULT_FILE  = 0

# Frame index to view; None = grid view, integer = single-frame triplet view
DEFAULT_FRAME = None   # e.g. 42

# Number of frames to show in grid view (used when DEFAULT_FRAME is None)
DEFAULT_N = 16
# ─────────────────────────────────────────────────────────────────────────────


def load_file(folder, file_idx):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npz')])
    if not files:
        raise FileNotFoundError(f"No npz files in {folder}")
    path = os.path.join(folder, files[file_idx % len(files)])
    d    = np.load(path)
    print(f"Loaded: {path}  |  {d['img'].shape[0]} frames")
    return d['img'], d['coll_img'], path


def show_single(img, coll_img, frame_idx, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{title}  |  frame {frame_idx}", fontsize=11)

    depth = img[frame_idx]
    ca    = coll_img[frame_idx]
    diff  = depth.astype(int) - ca.astype(int)   # positive = CA shows closer obstacles

    axes[0].imshow(depth, cmap='viridis', vmin=0, vmax=255)
    axes[0].set_title("Raw depth (img)")
    axes[0].axis('off')

    axes[1].imshow(ca, cmap='viridis', vmin=0, vmax=255)
    axes[1].set_title("CA image (coll_img)")
    axes[1].axis('off')

    im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-80, vmax=80)
    axes[2].set_title("Diff (img - coll_img)\nblue=CA closer, red=CA farther")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()


def show_grid(img, coll_img, n=16):
    """Show n randomly sampled frames; each column = raw depth | CA pair."""
    total   = img.shape[0]
    indices = sorted(np.random.choice(total, min(n, total), replace=False))

    cols = 8   # 4 pairs per row
    rows = int(np.ceil(len(indices) / (cols // 2)))

    fig, axes = plt.subplots(rows * 2, cols // 2, figsize=(cols * 1.6, rows * 3.4))
    axes = np.array(axes).reshape(rows * 2, cols // 2)

    for k, idx in enumerate(indices):
        row_pair = (k // (cols // 2)) * 2
        col      = k % (cols // 2)
        axes[row_pair    ][col].imshow(img[idx],      cmap='viridis', vmin=0, vmax=255)
        axes[row_pair    ][col].set_title(f"orig {idx}", fontsize=7)
        axes[row_pair    ][col].axis('off')
        axes[row_pair + 1][col].imshow(coll_img[idx], cmap='viridis', vmin=0, vmax=255)
        axes[row_pair + 1][col].set_title(f"CA {idx}",   fontsize=7)
        axes[row_pair + 1][col].axis('off')

    # hide unused grid cells
    for k in range(len(indices), rows * (cols // 2)):
        row_pair = (k // (cols // 2)) * 2
        col      = k % (cols // 2)
        axes[row_pair    ][col].axis('off')
        axes[row_pair + 1][col].axis('off')

    plt.suptitle(f"top=raw depth  bottom=CA  (random {len(indices)} frames)", fontsize=10)
    plt.tight_layout()
    plt.show()


def print_stats(img, coll_img):
    diff = img.astype(int) - coll_img.astype(int)
    print(f"\n{'':=<50}")
    print(f"  total frames    : {img.shape[0]}")
    print(f"  image size      : {img.shape[1]}x{img.shape[2]}  dtype={img.dtype}")
    print(f"  img      min/max/mean : {img.min():3d} / {img.max():3d} / {img.mean():.1f}")
    print(f"  coll_img min/max/mean : {coll_img.min():3d} / {coll_img.max():3d} / {coll_img.mean():.1f}")
    n_changed = np.sum(np.any(diff.reshape(img.shape[0], -1) != 0, axis=1))
    print(f"  frames with CA changes : {n_changed}/{img.shape[0]}  ({100*n_changed/img.shape[0]:.1f}%)")
    print(f"  diff mean/max          : {diff.mean():.2f} / {diff.max()}")
    print(f"{'':=<50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=CA_FOLDER,     help='CA npz directory')
    parser.add_argument('--file',   type=int, default=DEFAULT_FILE,  help='file index (0-based)')
    parser.add_argument('--frame',  type=int, default=DEFAULT_FRAME, help='frame index; omit for grid view')
    parser.add_argument('--n',      type=int, default=DEFAULT_N,     help='number of frames in grid view')
    args = parser.parse_args()

    img, coll_img, path = load_file(args.folder, args.file)
    print_stats(img, coll_img)

    if args.frame is not None:
        show_single(img, coll_img, args.frame, title=os.path.basename(path))
    else:
        show_grid(img, coll_img, n=args.n)
