import os
import numpy as np
import csv
from collections import Counter


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def find_row_by_name(filename, target_name):
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for perturbation in csv_reader:
            if perturbation['name'] == target_name:
                return perturbation
    return None
def disturb_matrices(perturbation_csv, target_name):

    perturbation = find_row_by_name(perturbation_csv, target_name)
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

def get_depth(calib_dir, velo_file_name, im_shape, perturb_path, name, cam=2, vel_depth=False, augmentation=None, extrinsic=False):
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    perturb_dir = os.path.dirname(perturb_path)

    # Equal to matrix Tr_velo_to_cam
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    if augmentation is not None:
        augmentation_csv = os.path.join(perturb_dir, augmentation)  # During eval
        rot_error, translation_error = disturb_matrices(perturbation_csv=augmentation_csv, target_name=name)
        velo2cam_augmented = np.dot(velo2cam, rot_error) + translation_error
    else:
        velo2cam_augmented = velo2cam

    # compute projection matrix velodyne -> image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam_augmented)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

    # depth information: 1st column of velo and 3rd column of velo_points_im
    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int_), velo_pts_im[:, 0].astype(np.int_)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    # Introducing Perturbation
    rot_error, translation_error = disturb_matrices(perturbation_csv=perturb_path, target_name=name)
    velo2cam_error = np.dot(velo2cam, rot_error) + translation_error

    # compute projection matrix velodyne->image plane
    P_velo2im2 = np.dot(np.dot(P_rect, R_cam2rect), velo2cam_error)

    # project the points to the camera
    velo_pts_im2 = np.dot(P_velo2im2, velo.T).T
    velo_pts_im2[:, :2] = velo_pts_im2[:, :2] / velo_pts_im2[:, 2][..., np.newaxis]

    # depth information (z): 1st column of velo and 3rd column of velo_points_im
    if vel_depth:
        velo_pts_im2[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im2[:, 0] = np.round(velo_pts_im2[:, 0]) - 1
    velo_pts_im2[:, 1] = np.round(velo_pts_im2[:, 1]) - 1
    val_inds2 = (velo_pts_im2[:, 0] >= 0) & (velo_pts_im2[:, 1] >= 0)
    val_inds2 = val_inds2 & (velo_pts_im2[:, 0] < im_shape[1]) & (velo_pts_im2[:, 1] < im_shape[0])
    velo_pts_im2 = velo_pts_im2[val_inds2, :]

    # project to image
    depth_neg = np.zeros((im_shape))
    depth_neg[velo_pts_im2[:, 1].astype(np.int_), velo_pts_im2[:, 0].astype(np.int_)] = velo_pts_im2[:, 2]

    # find the duplicate points and choose the closest depth
    inds2 = sub2ind(depth_neg.shape, velo_pts_im2[:, 1], velo_pts_im2[:, 0])
    dupe_inds2 = [item for item, count in Counter(inds2).items() if count > 1]
    for dd in dupe_inds2:
        pts = np.where(inds2 == dd)[0]
        x_loc = int(velo_pts_im2[pts[0], 0])
        y_loc = int(velo_pts_im2[pts[0], 1])
        depth_neg[y_loc, x_loc] = velo_pts_im2[pts, 2].min()
    depth_neg[depth_neg < 0] = 0

    return depth, depth_neg
