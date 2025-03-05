import pykitti
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# Initialize the dataset and DataLoader
current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../../'))
data_short_dir = 'data/kitti_odom'  # Replace with your actual data directory
datadir = os.path.join(root, data_short_dir)
phase = 'train'
perturb_filenames = 'perturbation_train.csv'  # Replace with your actual file name
batch_size = 1

dataset = pykitti.odometry(datadir, '00')

cam_index = '2'
frame_idx = 0000

T_cam_velo = getattr(dataset.calib, f"T_cam{cam_index}_velo")
P_rect = getattr(dataset.calib, f"P_rect_{cam_index}0")

# Load the image and Velodyne points for the given frame
rgb_image = list(getattr(dataset, f"cam{cam_index}"))[frame_idx]
velodyne_points = list(dataset.velo)[frame_idx]
im_shape = rgb_image.size
im_shape = im_shape[::-1]
print(im_shape)

full_transform = P_rect @ T_cam_velo  # Shape (3, 4)

# Convert Velodyne points to homogeneous coordinates
velo_hom = np.hstack((velodyne_points[:, :3], np.ones((velodyne_points.shape[0], 1))))

# Apply transformation
velo_pts_im = (full_transform @ velo_hom.T).T  # Shape (N, 3)
velo_pts_im[:, :2] /= velo_pts_im[:, 2][:, np.newaxis]  # Normalize

# Set the depth information directly
velo_pts_im[:, 2] = velo_hom[:, 0]

# Round coordinates and check bounds
velo_pts_im[:, :2] = np.round(velo_pts_im[:, :2]) - 1
val_inds = (
        (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 0] < im_shape[1]) &
        (velo_pts_im[:, 1] >= 0) & (velo_pts_im[:, 1] < im_shape[0])
)
velo_pts_im = velo_pts_im[val_inds]

# Initialize depth map
depth = np.zeros(im_shape)
depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[:, 2]

# Manage duplicates, keeping the closest depth
inds = np.ravel_multi_index((velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)), depth.shape)
dupe_inds = [idx for idx, count in Counter(inds).items() if count > 1]

for dd in dupe_inds:
    pts = np.where(inds == dd)[0]
    y_loc, x_loc = int(velo_pts_im[pts[0], 1]), int(velo_pts_im[pts[0], 0])
    depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

depth[depth < 0] = 0
print(im_shape)
print(depth.shape)

y_coords, x_coords = np.nonzero(depth > 0)
depth_values = depth[y_coords, x_coords]

# Scatter plot with depth as color
plt.scatter(x_coords, y_coords, c=depth_values, cmap='viridis', s=1, marker='.')
plt.colorbar(label='Depth (m)')
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.title('Depth Map Scatter Plot (Depth > 0)')
plt.xlabel('Width (pixels)')  # Corrected label
plt.ylabel('Height (pixels)')  # Corrected label
plt.show()