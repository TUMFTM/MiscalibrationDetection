import os
import numpy as np
import csv
from collections import Counter


def find_row_by_name(filename, target_name):
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for perturbation in csv_reader:
            if perturbation['name'] == target_name:
                return perturbation
    return None

def disturb_matrices(perturbation_csv, target_name):

    perturbation = find_row_by_name(perturbation_csv, target_name)
    assert perturbation is not None, "perturbation data is missing."

    theta_rad1 = float(perturbation["theta_rad1"])/180 * 2 * np.pi
    theta_rad2 = float(perturbation["theta_rad2"])/180 * 2 * np.pi
    theta_rad3 = float(perturbation["theta_rad3"])/180 * 2 * np.pi
    x = float(perturbation["x"])
    y = float(perturbation["y"])
    z = float(perturbation["z"])

    R1 = np.array([[np.cos(theta_rad1), -np.sin(theta_rad1), 0, 0],
                   [np.sin(theta_rad1), np.cos(theta_rad1), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    R2 = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta_rad2), -np.sin(theta_rad2), 0],
                   [0, np.sin(theta_rad2), np.cos(theta_rad2), 0],
                   [0, 0, 0, 1]])
    R3 = np.array([[np.cos(theta_rad3), 0, np.sin(theta_rad3), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta_rad3), 0, np.cos(theta_rad3), 0],
                   [0, 0, 0, 1]])
    rot_error = np.dot(R1, np.dot(R2, R3))

    translation_error = np.array([[0, 0, 0, x],
                                  [0, 0, 0, y],
                                  [0, 0, 0, z],
                                  [0, 0, 0, 0]])

    return rot_error, translation_error

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def project_velodyne_to_camera(velodyne_points, im_shape, T_cam_velo, P_rect, perturb_path, name, augmentation=None, intrinsic=False):
    """
    Projects Velodyne points onto the camera image plane and includes depth information.

    Args:
        velodyne_points (np.array): Array of Velodyne points (N, 4) where each point has (x, y, z, reflectance).
        T_cam_velo (np.array): Transformation matrix from Velodyne to camera frame.
        R_rect (np.array): Rectification matrix.
        P_rect (np.array): Projection matrix for the camera.

    Returns:
        np.array: Projected 2D points on the image plane with depth (M, 3), where each point has (x, y, z).
    """
    perturb_dir = os.path.dirname(perturb_path)

    if not intrinsic:
        R_error, T_error = disturb_matrices(perturbation_csv=perturb_path, target_name=name)
        T_cam_velo_err = T_cam_velo @ R_error + T_error

    if augmentation is not None:
        augmentation_csv = os.path.join(perturb_dir, augmentation)  # During eval
        rot_error, translation_error = disturb_matrices(perturbation_csv=augmentation_csv, target_name=name)
        T_cam_velo = np.dot(T_cam_velo, rot_error) + translation_error
    else:
        T_cam_velo = T_cam_velo

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
    # inds = np.ravel_multi_index((velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)), depth.shape)
    # dupe_inds = [idx for idx, count in Counter(inds).items() if count > 1]
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    ## Again for miscalibration
    # Apply extrinsic miscalibration
    if not intrinsic:
        # Apply transformation
        error_transform = P_rect @ T_cam_velo_err
        velo_pts_im_neg = (error_transform @ velo_hom.T).T  # Shape (N, 3)
        velo_pts_im_neg[:, :2] /= velo_pts_im_neg[:, 2][:, np.newaxis]  # Normalize

        # Set the depth information directly
        velo_pts_im_neg[:, 2] = velo_hom[:, 0]

        # Round coordinates and check bounds
        velo_pts_im_neg[:, :2] = np.round(velo_pts_im_neg[:, :2]) - 1
        val_inds_neg = (
                (velo_pts_im_neg[:, 0] >= 0) & (velo_pts_im_neg[:, 0] < im_shape[1]) &
                (velo_pts_im_neg[:, 1] >= 0) & (velo_pts_im_neg[:, 1] < im_shape[0])
        )
        velo_pts_im_neg = velo_pts_im_neg[val_inds_neg]

        # Initialize depth map
        depth_neg = np.zeros(im_shape)
        depth_neg[velo_pts_im_neg[:, 1].astype(int), velo_pts_im_neg[:, 0].astype(int)] = velo_pts_im_neg[:, 2]

        # Manage duplicates, keeping the closest depth
        # inds = np.ravel_multi_index((velo_pts_im_neg[:, 1].astype(int), velo_pts_im_neg[:, 0].astype(int)), depth_neg.shape)
        # dupe_inds = [idx for idx, count in Counter(inds).items() if count > 1]
        inds = sub2ind(depth_neg.shape, velo_pts_im_neg[:, 1], velo_pts_im_neg[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

        for dd in dupe_inds:
            pts = np.where(inds==dd)[0]
            x_loc = int(velo_pts_im_neg[pts[0], 0])
            y_loc = int(velo_pts_im_neg[pts[0], 1])
            depth_neg[y_loc, x_loc] = velo_pts_im_neg[pts, 2].min()

        depth_neg[depth_neg < 0] = 0
    else:
        print("Applying intrinsic miscalibration")
        
        perturbation_intr = find_row_by_name(filename=perturb_path, target_name=name)
        assert perturbation_intr is not None, "perturbation intrinsic data is missing."
        
        P_rect_err  = P_rect.copy()
        # P_rect_err[0,0] += float(perturbation_intr["fu"])
        # P_rect_err[1,1] += float(perturbation_intr["fv"])
        # P_rect_err[0,2] += float(perturbation_intr["cu"])
        # P_rect_err[1,2] += float(perturbation_intr["cv"])
        fu_err_percent = 1.0 + float(perturbation_intr["fu"])/100
        fv_err_percent = 1.0 + float(perturbation_intr["fv"])/100
        cu_err_percent = 1.0 + float(perturbation_intr["cu"])/100
        cv_err_percent = 1.0 + float(perturbation_intr["cv"])/100
        gamma_err_percent = float(perturbation_intr["gamma"])/100
        P_rect_err[0,0] *= fu_err_percent
        P_rect_err[1,1] *= fv_err_percent
        P_rect_err[0,2] *= cu_err_percent
        P_rect_err[1,2] *= cv_err_percent
        P_rect_err[0,1] += gamma_err_percent*P_rect[0,0]
        # P_rect_err[0,1] = float(perturbation_intr["gamma"])

        full_transform_intr = P_rect_err @ T_cam_velo

        # Apply transformation
        velo_pts_im_neg = (full_transform_intr @ velo_hom.T).T  # Shape (N, 3)
        velo_pts_im_neg[:, :2] /= velo_pts_im_neg[:, 2][:, np.newaxis]  # Normalize

        # Set the depth information directly
        velo_pts_im_neg[:, 2] = velo_hom[:, 0]

        # Round coordinates and check bounds
        velo_pts_im_neg[:, :2] = np.round(velo_pts_im_neg[:, :2]) - 1
        val_inds_neg = (
                (velo_pts_im_neg[:, 0] >= 0) & (velo_pts_im_neg[:, 0] < im_shape[1]) &
                (velo_pts_im_neg[:, 1] >= 0) & (velo_pts_im_neg[:, 1] < im_shape[0])
        )
        velo_pts_im_neg = velo_pts_im_neg[val_inds_neg]

        # Initialize depth map
        depth_neg = np.zeros(im_shape)
        depth_neg[velo_pts_im_neg[:, 1].astype(int), velo_pts_im_neg[:, 0].astype(int)] = velo_pts_im_neg[:, 2]

        # Manage duplicates, keeping the closest depth
        inds = sub2ind(depth_neg.shape, velo_pts_im_neg[:, 1], velo_pts_im_neg[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

        for dd in dupe_inds:
            pts = np.where(inds==dd)[0]
            x_loc = int(velo_pts_im_neg[pts[0], 0])
            y_loc = int(velo_pts_im_neg[pts[0], 1])
            depth_neg[y_loc, x_loc] = velo_pts_im_neg[pts, 2].min()

        depth_neg[depth_neg < 0] = 0

    return depth, depth_neg