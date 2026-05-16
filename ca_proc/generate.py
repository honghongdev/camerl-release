import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import sys
import io

import torch
import warp as wp
import trimesh as tm

from PIL import Image
from matplotlib import cm

import torch.nn as nn

from utils.depthConvert import depthImageToDepthMap, depthMapToDepthImage

from tqdm import tqdm

# Depth range (meters)
MAX_DEPTH = 10.0
MIN_DEPTH = 0.0

# Robot body size (meters): cube mesh edge length, controls CA "danger inflation" radius
ROBOT_EDGE_LENGTH = 0.2

# ── camera intrinsics (match AvoidBench simulation camera) ───────────────────
IMG_WIDTH  = 256
IMG_HEIGHT = 256
CX = IMG_WIDTH  / 2    # principal point x (pixels)
CY = IMG_HEIGHT / 2    # principal point y (pixels)
FX = 268.51644         # focal length x (horizontal FOV ≈ 100°)
FY = 268.51644         # focal length y
# ─────────────────────────────────────────────────────────────────────────────

IMG_FOLDER    = "./imgs/"
NPZ_SAVE_COUNT = 500


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.9, device=device)
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("Using CPU")


wp.init()

@wp.kernel
def draw(mesh: wp.uint64,
         cam_pos: wp.vec3,
         width: wp.int32,
         height: wp.int32,
         pixels: wp.array1d(dtype=wp.float32),
         cx: wp.float32,
         cy: wp.float32,
         fx: wp.float32,
         fy: wp.float32):

    tid = wp.tid()

    x = wp.float32(tid % width)
    y = wp.float32(tid // width)

    sx = float(x - cx) / fx
    sy = float(y - cy) / fy

    ro = cam_pos
    rd = wp.normalize(wp.vec3(sx, sy, 1.0))

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    color = 10.0

    if wp.mesh_query_ray(mesh, ro, rd, 50.0, t, u, v, sign, n, f):
        value = t * rd[2]
        if value < 0.2:
            color = -1.0
        else:
            color = t * rd[2]

    pixels[tid] = color


def create_meshgrid(height, width, cx, cy, fx, fy):
    """Creates a meshgrid.
    Parameters
    ----------
    height, width : int
    cx, cy        : float  principal point
    fx, fy        : float  focal lengths
    Returns
    -------
    np.ndarray  shape (3, H, W)
    """
    x = np.arange(0, height, dtype=np.float32)
    y = np.arange(0, width,  dtype=np.float32)
    x, y = np.meshgrid(y, x)
    z = np.ones((height, width))
    x = (x - cx) / fx
    y = (y - cy) / fy
    return np.stack([x, y, z], axis=0)


def depth_to_pointcloud(depth_img, meshgrid, scale=1.0, offset_dist=5.0):
    """Converts a depth image to a point cloud.
    Parameters
    ----------
    depth_img   : np.ndarray  (H, W) depth in meters
    meshgrid    : np.ndarray  (3, H, W) pre-computed meshgrid
    scale       : float
    offset_dist : float  robot half-size for z offset
    Returns
    -------
    point_cloud : np.ndarray (3, H, W)
    z_offset    : np.ndarray (H, W)
    """
    x = meshgrid[0] * depth_img * scale
    y = meshgrid[1] * depth_img * scale
    z = meshgrid[2] * depth_img * scale
    z_pcl    = z.copy()
    z_offset = z - offset_dist
    point_cloud = np.stack([x, y, z_pcl], axis=0)
    return point_cloud, z_offset


def create_cube_mesh(edges, point_cloud, edge_length=0.2):
    """
    Creates a cube mesh at the centres of detected edges.
    Parameters
    ----------
    edges       : np.ndarray  (N, 2) edge pixel coordinates
    point_cloud : np.ndarray  (3, H, W)
    edge_length : float  cube side length in meters
    Returns
    -------
    list of trimesh.Trimesh
    """
    num_edges = edges.shape[0]
    cube_mesh_list = []
    for i in range(num_edges):
        x_edge = edges[i, 1]
        y_edge = edges[i, 0]
        point_origin = point_cloud[:, y_edge, x_edge]
        cube_mesh_list.append(tm.creation.box(
            extents=[edge_length, edge_length, edge_length]))
        t = tm.creation.box(extents=[edge_length, edge_length, edge_length])
        t.apply_translation(point_origin)
        cube_mesh_list[-1].apply_translation(point_origin)
    return cube_mesh_list


def save_point_cloud(point_cloud, filename, edges=None):
    """Save point_cloud ([3, H, W]) to a .npz file (points: Nx3)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if edges is not None and edges.size > 0:
        pts = np.array([point_cloud[:, int(e[0]), int(e[1])] for e in edges])
    else:
        pts = point_cloud.reshape(3, -1).T
        pts = pts[pts[:, 2] > 0]
    np.savez_compressed(filename, points=pts)
    return pts


def visualize_point_cloud(pts, title="point cloud", save_path=None, show=True,
                           s=1, cmap='viridis', subsample=20000, clear_remote=False):
    """Simple matplotlib 3D scatter for point cloud (pts: Nx3)."""
    if clear_remote:
        dists = np.linalg.norm(pts, axis=1)
        pts   = pts[dists < 9.0]
    if pts is None or pts.size == 0:
        return
    if pts.shape[0] > subsample:
        idx      = np.random.choice(pts.shape[0], subsample, replace=False)
        pts_plot = pts[idx]
    else:
        pts_plot = pts

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
                     c=pts_plot[:, 2], cmap=cmap, s=s)
    try:
        ax.set_box_aspect([
            np.ptp(pts_plot[:, 0]) if np.ptp(pts_plot[:, 0]) > 0 else 1.0,
            np.ptp(pts_plot[:, 1]) if np.ptp(pts_plot[:, 1]) > 0 else 1.0,
            np.ptp(pts_plot[:, 2]) if np.ptp(pts_plot[:, 2]) > 0 else 1.0,
        ])
    except Exception:
        pass
    plt.colorbar(sc, ax=ax, label="Z")
    plt.tight_layout()
    ax.grid(False)
    ax.set_facecolor('none')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.w_xaxis.line.set_visible(False)
    ax.w_yaxis.line.set_visible(False)
    ax.w_zaxis.line.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=-45, azim=-90)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close(fig)


class CamParams:
    def __init__(self, cx=240, cy=135, fx=252.91646, fy=252.91646):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy


class ImageProcessor:
    """Normalizes depth values to [0, 1]."""
    def __init__(self, min_depth=0.20, max_depth=10.0, scaling_factor=0.10,
                 pixel_value_min_depth=-1.0, pixel_value_max_depth=10.0):
        self.min_depth              = min_depth
        self.max_depth              = max_depth
        self.pixel_value_min_depth  = pixel_value_min_depth
        self.pixel_value_max_depth  = pixel_value_max_depth
        self.scaling_factor         = scaling_factor

    def process_image(self, image):
        image[image < self.min_depth] = self.pixel_value_min_depth
        image[image > self.max_depth] = self.pixel_value_max_depth
        image = image * self.scaling_factor
        image[image < 0.0] = 0.0
        image[image > 1.0] = 1.0
        return image


class EdgeDetector:
    def __init__(self, threshold1=30, threshold2=50):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def process_image(self, image):
        """
        Detect edges via Canny and snap each edge pixel to its nearest-depth
        neighbour within a 2-pixel radius.

        Returns
        -------
        edges      : np.ndarray (N, 2)  refined edge coordinates
        edge_image : np.ndarray (H, W)  raw Canny output
        """
        edge_image = cv2.Canny(image, self.threshold1, self.threshold2)
        edges      = np.where(edge_image > 0)
        edges      = np.array(list(zip(edges[0], edges[1])))
        for i in range(edges.shape[0]):
            edge = edges[i]
            neighbor_list = [
                (max(edge[0] - 1, 0),             edge[1]),
                (min(edge[0] + 1, IMG_HEIGHT - 1), edge[1]),
                (edge[0], max(0, edge[1] - 1)),
                (edge[0], min(IMG_WIDTH - 1, edge[1] + 1)),
                (max(edge[0] - 2, 0),             edge[1]),
                (min(edge[0] + 2, IMG_HEIGHT - 1), edge[1]),
                (edge[0], max(0, edge[1] - 2)),
                (edge[0], min(IMG_WIDTH - 1, edge[1] + 2)),
            ]
            min_depth = image[edge[0], edge[1]]
            if min_depth <= 0.0:
                for j in neighbor_list:
                    if image[j[0], j[1]] > 0.0:
                        min_depth = image[j[0], j[1]]
                        edge      = j
                        break
            min_neighbor = (0, 0)
            for j in neighbor_list:
                if image[j[0], j[1]] < min_depth and image[j[0], j[1]] > MIN_DEPTH:
                    min_depth    = image[j[0], j[1]]
                    min_neighbor = j
            if min_depth < image[edge[0], edge[1]] and image[edge[0], edge[1]] > MIN_DEPTH:
                edges[i] = min_neighbor

        return edges, edge_image


class CollisionImageProcessor:
    def __init__(self, cam_params, edge_detector, image_processor):
        self.cam_params      = cam_params
        self.edge_detector   = edge_detector
        self.image_processor = image_processor
        self.meshgrid        = create_meshgrid(
            IMG_HEIGHT, IMG_WIDTH,
            cam_params.cx, cam_params.cy, cam_params.fx, cam_params.fy)

    def process_image(self, depth_img):
        point_cloud, offset_image = depth_to_pointcloud(
            depth_img, self.meshgrid, scale=1.0, offset_dist=ROBOT_EDGE_LENGTH / 2)

        save_point_cloud(point_cloud, "debug/point_cloud.npz")

        depth_img_processed = self.image_processor.process_image(depth_img.copy())

        edges, edge_image = self.edge_detector.process_image(
            (depth_img_processed * 255.0).astype(np.uint8))

        # early exit for open scenes with few edges
        if edges.ndim < 2 or len(edges) < 10:
            return None, None, None, None

        normalized_offset_image = self.image_processor.process_image(offset_image.copy())

        edge_depth_image = np.zeros_like(depth_img)
        edge_depth_image.fill(MAX_DEPTH)
        edge_depth_image[edges[:, 0], edges[:, 1]] = \
            depth_img_processed[edges[:, 0], edges[:, 1]] * MAX_DEPTH
        plt.imsave("debug/edge_depth_image.png", edge_depth_image,
                   cmap='viridis', vmin=0, vmax=10)

        edge_pc, _ = depth_to_pointcloud(
            edge_depth_image, self.meshgrid, scale=1.0, offset_dist=ROBOT_EDGE_LENGTH / 2)
        save_point_cloud(edge_pc, "debug/edge_point_cloud.npz")

        edges = edges[::5]  # subsample edges for mesh construction

        cube_mesh_list       = create_cube_mesh(edges, point_cloud, edge_length=ROBOT_EDGE_LENGTH)
        cube_mesh_aggregated = tm.util.concatenate(cube_mesh_list)

        points  = wp.array(np.array(cube_mesh_aggregated.vertices), dtype=wp.vec3,  device="cuda:0")
        faces   = wp.array(np.array(cube_mesh_aggregated.faces.flatten()), dtype=wp.int32, device="cuda:0")
        wp_mesh = wp.Mesh(points, faces)

        pixels = wp.zeros(IMG_HEIGHT * IMG_WIDTH, dtype=wp.float32, device="cuda:0")
        wp.launch(
            kernel=draw,
            dim=IMG_HEIGHT * IMG_WIDTH,
            inputs=[wp_mesh.id, wp.vec3(0, 0, 0),
                    IMG_WIDTH, IMG_HEIGHT, pixels,
                    self.cam_params.cx, self.cam_params.cy,
                    self.cam_params.fx, self.cam_params.fy],
        )

        raycast_img             = pixels.numpy().reshape(IMG_HEIGHT, IMG_WIDTH)
        normalized_raycast_image = self.image_processor.process_image(raycast_img.copy())

        combined_collision_image = np.minimum(normalized_offset_image, normalized_raycast_image)
        return combined_collision_image, normalized_raycast_image, normalized_offset_image, edge_image


def main():
    parser = argparse.ArgumentParser(description='CA preprocessing: convert depth rollouts to CA maps')
    parser.add_argument('--input',  type=str, default='../saved/dataset/',
                        help='Directory containing raw rollout_*.npz files')
    parser.add_argument('--output', type=str, default='../saved/dataset-ca/',
                        help='Output directory for CA npz files')
    args = parser.parse_args()

    NPZ_FOLDER       = args.input
    NPZ_SAVE_FOLDER  = args.output
    COLL_IMG_SAVE_FOLDER = os.path.join(NPZ_SAVE_FOLDER, 'samples/')

    os.makedirs(NPZ_SAVE_FOLDER,      exist_ok=True)
    os.makedirs(COLL_IMG_SAVE_FOLDER, exist_ok=True)
    os.makedirs("debug",              exist_ok=True)

    cam_params             = CamParams(CX, CY, FX, FY)
    image_processor        = ImageProcessor(0.2, MAX_DEPTH, 0.1, -1.0, MAX_DEPTH)
    edge_detector          = EdgeDetector(30, 50)
    collision_image_processor = CollisionImageProcessor(cam_params, edge_detector, image_processor)

    npz_files = sorted([os.path.join(NPZ_FOLDER, f)
                        for f in os.listdir(NPZ_FOLDER) if f.endswith('.npz')])
    n_files   = len(npz_files)
    print(f"Found {n_files} npz files in {NPZ_FOLDER}")

    print("Counting total frames...")
    frame_counts = []
    for f in npz_files:
        with np.load(f, allow_pickle=True) as d:
            frame_counts.append(d['observations'].item()['image'].shape[0])
    total_frames = sum(frame_counts)
    print(f"Total frames to process: {total_frames}")

    img_save_count = 0

    file_bar  = tqdm(total=n_files,        desc="Files",  unit="file", position=0)
    frame_bar = tqdm(total=total_frames,   desc="Frames", unit="img",  position=1,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for batch_idx, file in enumerate(npz_files):
        npz_data = np.load(file, allow_pickle=True)
        imgs     = npz_data['observations'].item()['image'].astype(np.uint8)
        n_frames = imgs.shape[0]

        file_bar.set_description(
            f"File {batch_idx+1}/{n_files}: {os.path.basename(file)} ({n_frames} frames)")

        out_npz_list = []
        for i in range(n_frames):
            img = imgs[i]
            img = depthImageToDepthMap(img.squeeze(0), d_min=MIN_DEPTH, d_max=MAX_DEPTH)
            depth_data_torch = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to("cpu").float()
            depth_data_torch /= MAX_DEPTH
            depth_data_torch[depth_data_torch < MIN_DEPTH / MAX_DEPTH] = 0.0
            depth_data_torch[depth_data_torch > 1.0] = 1.0

            depth_data_torch_true_depth = depth_data_torch.clone() * MAX_DEPTH
            depth_data_torch_true_depth[depth_data_torch_true_depth < MIN_DEPTH] = -1.0

            processed_image, normalized_raycast_image, normalized_offset_image, edge_image = \
                collision_image_processor.process_image(
                    depth_data_torch_true_depth[0].detach().numpy().squeeze(0))

            if processed_image is None:
                origin_img = depthMapToDepthImage(
                    depth_data_torch_true_depth[0, 0].numpy(), d_min=MIN_DEPTH, d_max=MAX_DEPTH)
                out_npz_list.append((origin_img, origin_img))
                frame_bar.update(1)
                continue

            processed_collision_image_full_depth = processed_image * MAX_DEPTH
            processed_collision_image_full_depth[
                processed_collision_image_full_depth < MIN_DEPTH] = 0.0

            if img_save_count < COLL_IMG_SAVE_COUNT:
                combined_np  = np.concatenate(
                    (processed_collision_image_full_depth,
                     depth_data_torch_true_depth[0, 0].numpy()), axis=1)
                combined_img = Image.fromarray(combined_np)
                plt.imsave(
                    COLL_IMG_SAVE_FOLDER + "combined_img_" + str(img_save_count) + ".png",
                    combined_img, cmap='viridis', vmin=0, vmax=10)

            img_save_count += 1
            origin_img = depthMapToDepthImage(
                depth_data_torch_true_depth[0, 0].numpy(), d_min=MIN_DEPTH, d_max=MAX_DEPTH)
            coll_img   = depthMapToDepthImage(
                processed_collision_image_full_depth, d_min=MIN_DEPTH, d_max=MAX_DEPTH)
            out_npz_list.append((origin_img, coll_img))

            frame_bar.update(1)

        out_npz_list = np.array(out_npz_list)
        save_path    = NPZ_SAVE_FOLDER + "collision_images_{:03d}.npz".format(batch_idx)
        np.savez_compressed(save_path,
                            img=out_npz_list[:, 0],
                            coll_img=out_npz_list[:, 1])
        file_bar.update(1)

    file_bar.close()
    frame_bar.close()
    print(f"\nDone. Processed {img_save_count} frames, saved to {NPZ_SAVE_FOLDER}")


def process_one_img(img_path="./imgs/img_054.png"):
    """Process a single depth image and save debug visualizations to ./debug/."""
    cam_params             = CamParams(CX, CY, FX, FY)
    image_processor        = ImageProcessor(0.2, MAX_DEPTH, 0.1, -1.0, MAX_DEPTH)
    edge_detector          = EdgeDetector(30, 50)
    collision_image_processor = CollisionImageProcessor(cam_params, edge_detector, image_processor)

    depth_img = np.array(Image.open(img_path)).astype(np.uint8)
    depth_img = depthImageToDepthMap(depth_img, d_min=MIN_DEPTH, d_max=MAX_DEPTH)

    processed_image, normalized_raycast_image, normalized_offset_image, edge_image = \
        collision_image_processor.process_image(depth_img)

    processed_collision_image_full_depth = processed_image * MAX_DEPTH
    processed_collision_image_full_depth[processed_collision_image_full_depth < MIN_DEPTH] = 0.0

    os.makedirs("./debug", exist_ok=True)

    combined_np  = np.concatenate((
        processed_collision_image_full_depth, depth_img, edge_image,
        normalized_raycast_image * MAX_DEPTH, normalized_offset_image * MAX_DEPTH), axis=1)
    combined_img = Image.fromarray(combined_np)
    plt.imsave("./debug/combined_img.png", combined_img, cmap='viridis',
               vmin=MIN_DEPTH, vmax=MAX_DEPTH)

    def normalize_to_uint8(img, bias=True):
        if bias:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 230 + 25
        else:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        return img.astype(np.uint8)

    plt.imsave("./debug/coll_img_color.png",
               normalize_to_uint8(processed_collision_image_full_depth, False), cmap='viridis')
    plt.imsave("./debug/origin_img_color.png",    normalize_to_uint8(depth_img),                        cmap='viridis')
    plt.imsave("./debug/raycast_img_color.png",   normalize_to_uint8(normalized_raycast_image * MAX_DEPTH), cmap='viridis')
    plt.imsave("./debug/offset_img_color.png",    normalize_to_uint8(normalized_offset_image * MAX_DEPTH, False), cmap='viridis')
    plt.imsave("./debug/edge_img_color.png",      normalize_to_uint8(edge_image),                       cmap='viridis')
    plt.imsave("./debug/combined_img_color.png",
               np.concatenate((
                   normalize_to_uint8(processed_collision_image_full_depth),
                   normalize_to_uint8(depth_img),
                   normalize_to_uint8(edge_image),
                   normalize_to_uint8(normalized_raycast_image * MAX_DEPTH),
                   normalize_to_uint8(normalized_offset_image * MAX_DEPTH),
               ), axis=1), cmap='viridis')


if __name__ == "__main__":
    main()
    # process_one_img()
    print("Done.")
    sys.exit(0)
