import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# ============================
#  Define Rotation Matrices
# ============================

def rotation_x(theta):
    """Rotation matrix around the x-axis."""
    return tf.convert_to_tensor([
        [1, 0, 0, 0],
        [0, tf.cos(theta), -tf.sin(theta), 0],
        [0, tf.sin(theta), tf.cos(theta), 0],
        [0, 0, 0, 1]
    ], dtype=tf.float32)

def rotation_y(theta):
    """Rotation matrix around the y-axis."""
    return tf.convert_to_tensor([
        [tf.cos(theta), 0, tf.sin(theta), 0],
        [0, 1, 0, 0],
        [-tf.sin(theta), 0, tf.cos(theta), 0],
        [0, 0, 0, 1]
    ], dtype=tf.float32)

def rotation_z(theta):
    """Rotation matrix around the z-axis."""
    return tf.convert_to_tensor([
        [tf.cos(theta), -tf.sin(theta), 0, 0],
        [tf.sin(theta), tf.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=tf.float32)

# ============================
#  Apply Transformation Error
# ============================

def apply_error_to_transform(T, disturb=True):
    """
    Applies rotation and translation error to a given transformation matrix T.

    Args:
        T: Tensor (4,4) representing the transformation matrix.
        disturb: Boolean. If True, applies rotation and translation errors.

    Returns:
        Modified transformation matrix (4,4) with errors applied.
    """
    if not disturb:
        return T  # Return original transformation without disturbance

    # Define rotation errors (modify these values as needed)
    theta_1_err = tf.constant(0.1, dtype=tf.float32)  # Small rotation error in radians
    theta_2_err = tf.constant(0.0, dtype=tf.float32)
    theta_3_err = tf.constant(0.0, dtype=tf.float32)

    # Compute the total rotation error matrix R_err = R1_err * R2_err * R3_err
    R1_err = rotation_x(theta_1_err)
    R2_err = rotation_y(theta_2_err)
    R3_err = rotation_z(theta_3_err)

    R_total_err = tf.linalg.matmul(R1_err, tf.linalg.matmul(R2_err, R3_err))

    # Define translation error matrix
    t_err = tf.convert_to_tensor([
        [0, 0, 0, 1],  # Small translation error in x
        [0, 0, 0, 0],  # Small translation error in y
        [0, 0, 0, 0],  # Small translation error in z
        [0, 0, 0, 0]
    ], dtype=tf.float32)

    # Apply error transformation
    
    T_new = tf.linalg.matmul(T, R_total_err) + t_err
    return T_new

# ============================
#  Convert Range Image to Cartesian
# ============================

def convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False,
                                     disturb=False):
    """Convert range images from polar to Cartesian coordinates with optional disturbance."""
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)

    # Convert pose tensor to (H, W, 4, 4) without disturbance
    range_image_top_pose_tensor = transform_utils.get_transform(
        transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2]),
        range_image_top_pose_tensor[..., 3:])

    for c in frame.context.laser_calibrations:
        print(f"Processing {c.name} lidar...")
        range_image = range_images[c.name][ri_index]
        
        if len(c.beam_inclinations) == 0:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])

        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
        extrinsic = tf.convert_to_tensor(extrinsic, dtype=tf.float32)

        if disturb:
            extrinsic = apply_error_to_transform(extrinsic, disturb)

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)

        pixel_pose_local = None
        frame_pose_local = None

        if c.name == open_dataset.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)

        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0), 
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        if keep_polar_features:
            # If we want to keep the polar coordinate features of range, intensity,
            # and elongation, concatenate them to be the initial dimensions of the
            # returned Cartesian range image.
            range_image_cartesian = tf.concat(
                [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

        cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images




def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False,
                                       disturb=False):
    """Convert range images to point cloud.

    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
        camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.
        keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        (NOTE: Will be {[N, 6]} if keep_polar_features is true.
        cp_points: {[N, 6]} list of camera projections of length 5
        (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []

    cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features, disturb=disturb)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian,
                                    tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor,
                                        tf.compat.v1.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

    return points, cp_points

def project_lidar_to_image(points, camera_projection):
    """
    Projects 3D LiDAR points onto a 2D camera image using the camera projection matrix.

    Args:
        points: (N, 3) NumPy array of 3D LiDAR points (x, y, z).
        camera_projection: (N, 6) NumPy array containing camera projection info.
        
    Returns:
        projected_points: (M, 3) NumPy array of projected points with (u, v, z).
    """
    # Extract the 2D pixel coordinates (u, v) from the projection matrix
    u = camera_projection[:, 1]  # X (horizontal)
    v = camera_projection[:, 2]  # Y (vertical)
    
    # Extract the Z values for depth-based coloring
    z_values = points[:, 2]  # Use Z-coordinates as depth (color intensity)
    z_values[z_values < 0] = 0  # Clip negative values to 0

    # Filter out points that are outside the image bounds (optional)
    img_width, img_height = 1920, 1280  # Adjust based on your dataset
    mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)

    return np.stack([u[mask], v[mask], z_values[mask]], axis=1)

# ============================
#  Process and Visualize Data
# ============================

if __name__ == "__main__":
    # Path to the TFRecord file
    # Path to the TFRecord file
    tfrecord_path = "data/wod/training/testing_0000/segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord"

    # Load the dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        break

    # Parse range images and camera projections
    (range_images, camera_projections, _, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    disturb = True
    # Convert range images to point clouds
    points, cp_points = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, disturb=disturb
    )

    # Combine points from all 5 LiDARs into one array
    points_all = np.concatenate(points, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)

    # Select only points projected to the front camera
    front_camera_name = open_dataset.CameraName.FRONT
    mask = cp_points_all[:, 0] == front_camera_name
    print(f"frotn_camera_name: {front_camera_name}")
    print(f"shape of cp_points_all: {cp_points_all.shape}")

    # Extract only the points projected onto the front camera
    cp_points_front = cp_points_all[mask]
    points_front = points_all[mask]

    # Calculate distance for color mapping
    distances = np.linalg.norm(points_front, axis=-1)

    # Get the front camera image
    front_image = None
    for image in frame.images:
        if image.name == front_camera_name:
            front_image = tf.image.decode_jpeg(image.image).numpy()
            break

    # Plot the front camera image with LiDAR points
    plt.figure(figsize=(12, 7))
    plt.imshow(front_image)
    plt.title("Front Camera View with LiDAR Projection")

    # Plot the LiDAR points projected onto the camera view
    plt.scatter(cp_points_front[:, 1], cp_points_front[:, 2], c=distances, cmap='jet', s=2, alpha=0.6)
    plt.colorbar(label="Distance (m)")

    # Save the output image
    plt.axis('off')
    if disturb:
        plt.savefig("front_camera_with_lidar_projection_disturb.png", dpi=300)
    else:
        plt.savefig("front_camera_with_lidar_projection_nodisturb.png", dpi=300)

    # 🔹 Get the front camera projection matrix
    front_camera_name = open_dataset.CameraName.FRONT
    mask = cp_points_all[:, 0] == front_camera_name
    projected_lidar = project_lidar_to_image(points_all[mask], cp_points_all[mask])
    

    # 🔹 Plot the camera image
    plt.figure(figsize=(12, 7))
    plt.imshow(front_image)  # Background image
    plt.title("Projected LiDAR Points on Camera Image")

    # 🔹 Overlay LiDAR points on the image
    plt.scatter(
        projected_lidar[:, 0], projected_lidar[:, 1], 
        c=projected_lidar[:, 2], cmap="jet", s=2, alpha=0.6
    )
    plt.colorbar(label="Distance (Z in meters)")  # Color by depth (Z)

    # 🔹 Save or display the image
    plt.axis("off")
    plt.savefig("front_camera_with_lidar_heatmap.png", dpi=300)

