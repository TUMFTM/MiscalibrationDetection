import numpy as np
import open3d as o3d
import os

# Function to load point cloud data from a .bin file
def load_velodyne_bin_file(bin_file_path):
    # Velodyne point clouds are stored in a binary file with 4 floats per point (x, y, z, intensity)
    point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

# Load the point cloud data from a .bin file
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_file_path = os.path.join(root, "./data/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000004.bin")
point_cloud_data = load_velodyne_bin_file(bin_file_path)

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()

# Convert the numpy array to an Open3D point cloud
pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

# Optionally, add colors or other attributes here
# For example, set colors based on intensity (this is just an example)
colors = np.zeros_like(point_cloud_data[:, :3])
colors[:, 0] = point_cloud_data[:, 3]  # Use intensity for coloring
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save as PNG option
def save_point_cloud_as_png(pcd, file_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Open window in the background
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_path)
    vis.destroy_window()
    print(f"Point cloud saved as PNG at {file_path}")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# # Option to save the point cloud as a .png image
# save_option = input("Would you like to save the point cloud as an image? (y/n): ").strip().lower()
# if save_option == 'y':
#     save_path = os.path.join(root, "./outputs/others/point_cloud.png")
#     save_point_cloud_as_png(pcd, save_path)



