import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Alternative: 'Qt5Agg' or 'GTK3Agg'
import matplotlib.pyplot as plt


from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

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

    # Define translation error matrix (modify if needed)
    t_err = tf.convert_to_tensor([
        [0, 0, 0, 1],  # Small translation error in x
        [0, 0, 0, 0],  
        [0, 0, 0, 0],  
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
            print(f"Processing {c.name} lidar local...")
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

        points.append(points_tensor.numpy())

    return points
    

# ============================
#  Process and Visualize Data
# ============================

if __name__ == "__main__":
    # Replace with your own TFRecord path
    tfrecord_path = "data/wod/training/testing_0000/segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord"
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    # Load a single frame for demonstration
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        break

    # Parse range images (we don't necessarily need camera projections here, 
    # but parse_range_image_and_camera_projection returns them by default)
    range_images, *_ , range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    
    # Get LiDAR points (optionally with some artificial disturbance to the extrinsic)
    disturb = False
    points = convert_range_image_to_point_cloud(
        frame, range_images, range_image_top_pose, disturb=disturb
    )
    
    # Combine all LiDAR returns into a single Nx3 array
    points_all = np.concatenate(points, axis=0)
    # top_laser_name = open_dataset.LaserName.TOP
    # points_all = points[top_laser_name]

    # Extract X, Y, Z coordinates
    x = points_all[:, 0]
    y = points_all[:, 1]
    z = points_all[:, 2]

    # Plot 3D point cloud
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c=z, cmap='jet', alpha=0.5)

    # Labels and title
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("Waymo Open Dataset - LiDAR Point Cloud")

    # Show the plot
    plt.show(block=True)

    # Find the front camera RGB image in the frame
    front_camera_name = open_dataset.CameraName.FRONT
    front_image = None
    for image in frame.images:
        if image.name == front_camera_name:
            front_image = tf.image.decode_jpeg(image.image).numpy()
            break

    # -------------------------------------------------------------------------
    # TODO IMPLEMENTATION: project points from (x, y, z) in vehicle frame
    #                      to (u, v, z) in the Front Camera image.
    # -------------------------------------------------------------------------

    # 1. Get the camera calibration for the FRONT camera
    front_cam_calib = None
    for c in frame.context.camera_calibrations:
        if c.name == front_camera_name:
            front_cam_calib = c
            break
    
    projected_points = project_vehicle_to_image(frame.pose, front_cam_calib, points_all)
    print(projected_points.shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(front_image)
    plt.scatter(projected_points[0], projected_points[1], s=2, c='red', alpha=0.5)
    plt.title("LiDAR Points Projected onto Front Camera")
    plt.xlabel("u (px)")
    plt.ylabel("v (px)")
    plt.savefig("front_camera_with_lidar_projection_disturb.png", dpi=300)



































    # if front_cam_calib is None:
    #     raise ValueError("Front camera calibration not found in frame.")

    # # 2. Extract intrinsic parameters (row-major in c.intrinsic)
    # #    The layout is [fx, 0, cx,  0, fy, cy,  0, 0, 1]
    # intrinsic_cam = front_cam_calib.intrinsic
    # fx, fy = intrinsic_cam[0], intrinsic_cam[1]   # f_x, f_y
    # cx, cy = intrinsic_cam[2], intrinsic_cam[3]   # c_x, c_y
    # intrinsic_cam_matrix = np.array([
    #     [fx, 0, cx, 0],
    #     [0, fy, cy, 0],
    #     [0, 0, 1, 0]
    # ])

    # image_width  = front_cam_calib.width
    # image_height = front_cam_calib.height

    # # 3. Extract extrinsic: transforms points from vehicle frame to camera frame
    # extrinsic_cam = np.array(front_cam_calib.extrinsic.transform).reshape(4,4)

    # # 4. Convert points_all from Nx3 -> Nx4 homogeneous
    # ones = np.ones((points_all.shape[0], 1), dtype=np.float32)
    # points_hom = np.concatenate([points_all[:, :3], ones], axis=-1)  # Nx4

    # # 5. Transform to camera frame
    # #    p_cam = T_cam_vehicle * p_vehicle
    # points_cam = (extrinsic_cam @ points_hom.T).T  # Nx4

    # # 6. Keep points with z_cam > 0
    # mask_in_front = points_cam[:, 2] > 0
    # points_cam = points_cam[mask_in_front]

    # # 7. Perspective projection
    # x_cam = points_cam[:, 0]
    # y_cam = points_cam[:, 1]
    # z_cam = points_cam[:, 2]

    # u = fx * (x_cam / z_cam) + cx
    # v = fy * (y_cam / z_cam) + cy

    # # 8. Filter for points that fall within the image boundaries
    # mask_in_img = (
    #     (u >= 0) & (u < image_width) &
    #     (v >= 0) & (v < image_height)
    # )

    # u_valid = u[mask_in_img]
    # v_valid = v[mask_in_img]
    # z_valid = z_cam[mask_in_img]

    # # (Optionally) create an array holding (u,v,z) if you need them
    # uvz = np.stack([u_valid, v_valid, z_valid], axis=-1)

    # # -------------------------------------------------------------------------
    # # Visualization: overlay the projected points on the Front camera image
    # # -------------------------------------------------------------------------
    # plt.figure(figsize=(10, 8))
    # plt.imshow(front_image)
    # plt.scatter(u_valid, v_valid, s=2, c='red', alpha=0.5)
    # plt.title("LiDAR Points Projected onto Front Camera")
    # plt.xlabel("u (px)")
    # plt.ylabel("v (px)")
    # plt.savefig("front_camera_with_lidar_projection_disturb.png", dpi=300)
