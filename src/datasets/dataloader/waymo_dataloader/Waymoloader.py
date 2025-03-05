import numpy as np
from torch_waymo import WaymoDataset
import torch

# WARNING: This feature is currently not implemented.
# Future implementation will include the waymo dataloader

# Function to project LiDAR points to 2D image space with depth information
def project_lidar_to_image_with_depth(lidar_points, T_cam_lidar, K):
    """
    Project 3D LiDAR points to 2D camera image coordinates and return depth information.

    :param lidar_points: (N, 3) array of 3D points in LiDAR space
    :param T_cam_lidar: (4, 4) transformation matrix from LiDAR to camera frame
    :param K: (3, 3) camera intrinsic matrix
    :return: (N, 2) array of 2D image coordinates and (N,) array of depths
    """

    # Convert lidar points to homogeneous coordinates (N, 4)
    num_points = lidar_points.shape[0]
    lidar_points_hom = np.hstack((lidar_points, np.ones((num_points, 1))))

    # Transform lidar points to camera frame using T_cam_lidar
    cam_points_hom = T_cam_lidar @ lidar_points_hom.T  # (4, N)

    # Get depth (Z coordinate in the camera frame)
    depths = cam_points_hom[2, :]  # (N,)

    # Only keep points in front of the camera (Z > 0)
    valid_mask = depths > 0
    cam_points_hom = cam_points_hom[:, valid_mask]
    depths = depths[valid_mask]

    # Project to 2D using the intrinsic matrix
    cam_points_2d = K @ cam_points_hom[:3, :]  # (3, N)

    # Normalize by depth (Z) to get pixel coordinates
    cam_points_2d[:2, :] /= cam_points_2d[2, :]

    # Return the 2D image coordinates (u, v) and corresponding depths
    return cam_points_2d[:2, :].T, depths  # (N, 2), (N,)


# Example setup for using WaymoDataset
waymo_data_path = "/path/to/waymo/dataset"
dataset = WaymoDataset(root_dir=waymo_data_path, split="train", lidar=True, camera=True)

# Example to fetch a batch of data
for data in dataset:
    lidar_data = data['lidar']  # Get 3D LiDAR points (N, 3)
    camera_intrinsics = data['camera_intrinsics']  # Get camera intrinsic matrix (3, 3)
    camera_extrinsics = data['camera_extrinsics']  # Get camera to LiDAR transformation matrix (4, 4)

    # Assuming lidar_data is a Tensor, convert it to numpy for this example
    lidar_points = lidar_data.numpy()

    # Apply the projection from 3D to 2D with depth information
    projected_points, depths = project_lidar_to_image_with_depth(lidar_points, camera_extrinsics, camera_intrinsics)

    # `projected_points` contains the 2D (u, v) image coordinates
    # `depths` contains the corresponding depth values
    print("Projected 2D points: ", projected_points)
    print("Corresponding depths: ", depths)
    break  # Use the first batch for simplicity
