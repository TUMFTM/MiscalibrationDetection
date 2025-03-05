import pykitti
import os

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    base_dir = os.path.join(root, 'data/kitti_odom')
    sequence = "00"
    dataset = pykitti.odometry(base_dir, sequence)
    T_cam_velo = getattr(dataset.calib, f"T_cam2_velo")

    print(T_cam_velo)