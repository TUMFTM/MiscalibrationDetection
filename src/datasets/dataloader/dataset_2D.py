from src.datasets.dataloader.kitti_dataloader import Kittiloader
from src.datasets.dataloader.Transformer.custom_transformer import CustTransformer
from torch.utils.data import Dataset, DataLoader
import pykitti
from src.datasets.dataloader.kitti_odom_dataloader.bin2depth import *
import sys
from PIL import Image



class KittiDataset(Dataset):
    def __init__(self,
                 kittiDir,
                 mode,
                 perturb_filenames,
                 transform=None,
                 augmentation=None,
                 extrinsic=False):
        self.mode = mode
        self.kitti_root = kittiDir
        self.perturb_filenames = perturb_filenames
        self.transform = transform
        self.augmentation = augmentation
        self.extrinsic = extrinsic

        # use left image by default
        if self.augmentation is not None:
            print(f"Create augmentation for correct input. {self.augmentation}")
        else:
            print(f"No augmentation for correct input. {self.augmentation}")
            
        self.kittiloader = Kittiloader(kittiDir, mode, perturb_filenames, cam=2, augmentation=self.augmentation, extrinsic=self.extrinsic)

    def __getitem__(self, idx):
        # load an item according to the given index
        data_item = self.kittiloader.load_item(idx)     # get a set of data
        data_transed = self.transform(data_item)        # transform the set of data with CustTransformer.get_transform()

        return data_transed

    def __len__(self):
        return self.kittiloader.data_length()

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class KITTIOdometryDataset(Dataset):
    def __init__(self, datadir, phase, perturb_filenames, cam_index=2, transform=None, augmentation=None, intrinsic=False):
        """
        Args:
            datadir (str): Path to the KITTI odometry dataset.
            phase (str): Phase of dataset (e.g., 'train', 'test').
            perturb_filenames (str): Filename for perturbation csv.
            cam_index (int): Camera index to project points onto (0, 1, 2, or 3).
            transform (callable, optional): Transformation to apply to images and points.
            augmentation (optional): Augmentation to apply for perturbations.
            intrinsic (bool): Whether to use intrinsic matrix for projection. Otherwise, use extrinsic matrix.
        """
        self.basedir = datadir
        self.cam_index = cam_index
        self.transform = transform
        self.augmentation = augmentation
        self.intrinsic = intrinsic

        # use left image by default
        if self.augmentation is not None:
            print(f"Create augmentation for correct input. {self.augmentation}")
        else:
            print(f"No augmentation for correct input. {self.augmentation}")

        sequence_list_file = os.path.join(datadir, f'sequence_list_{phase}.txt')
        self.perturb_path = os.path.join(datadir, perturb_filenames)

        self.data_indices = []
        with open(sequence_list_file, 'r') as f:
            sequences_file = [line.strip() for line in f if line.strip()]

            for file_name in sequences_file:
                # Split by underscore to get sequence and idx
                sequence, idx = file_name.split('_')
                # Append the (sequence, idx) tuple to data_indices
                self.data_indices.append((sequence, int(idx)))

        sequence_folder = os.path.join(datadir, f'sequence_folder_{phase}.txt')
        self.transform_matrices_dict = {}
        with open(sequence_folder, 'r') as f:
            sequence_list = [line.strip() for line in f if line.strip()]

            for sequence in sequence_list:
                with SuppressPrint():
                    dataset = pykitti.odometry(self.basedir, sequence)
                    T_cam_velo = getattr(dataset.calib, f"T_cam{self.cam_index}_velo")
                    P_rect = getattr(dataset.calib, f"P_rect_{self.cam_index}0")

                    self.transform_matrices_dict[sequence] = {
                        "T_cam_velo": T_cam_velo,
                        "P_rect": P_rect
                    }

    def __len__(self):
        return len(self.data_indices)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.basedir, filename)
        assert os.path.exists(file_path), err_info
        return file_path

    def _load_velodyne_points(self, file_name):
        # adapted from https://github.com/hunse/kitti
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # homogeneous
        return points

    def __getitem__(self, idx):
        # Retrieve the sequence and frame index
        sequence, frame_idx = self.data_indices[idx]

        # Load the specific sequence dataset and calibration parameters
        # with SuppressPrint():
        #     dataset = pykitti.odometry(self.basedir, sequence)

        # T_cam_velo = getattr(dataset.calib, f"T_cam{self.cam_index}_velo")
        # P_rect = getattr(dataset.calib, f"P_rect_{self.cam_index}0")

        T_cam_velo = self.transform_matrices_dict[sequence]["T_cam_velo"]
        P_rect = self.transform_matrices_dict[sequence]["P_rect"]

        rgb_path = os.path.join(self.basedir, f"sequences/{sequence}/image_{self.cam_index}/{frame_idx:06d}.png")
        velo_path = os.path.join(self.basedir, f"sequences/{sequence}/velodyne/{frame_idx:06d}.bin")

        # Load the image and Velodyne points for the given frame
        # rgb_image = list(getattr(dataset, f"cam{self.cam_index}"))[frame_idx]
        # velodyne_points = list(dataset.velo)[frame_idx]

        rgb_image = Image.open(rgb_path).convert('RGB')
        velodyne_points = self._load_velodyne_points(velo_path)

        name = f"{sequence}_{frame_idx:06d}"

        # Dynamically set im_shape based on rgb_image dimensions
        im_shape = rgb_image.size  # (width, height)
        im_shape = im_shape[::-1]

        # Project Velodyne points onto the image
        depth, depth_neg = project_velodyne_to_camera(
            velodyne_points, im_shape, T_cam_velo, P_rect, self.perturb_path,
            name, augmentation=self.augmentation, intrinsic=self.intrinsic
        )

        # Create data item with a zero-padded name
        data_item = {
            'left_img': rgb_image.convert('RGB'),  # Ensure RGB format if necessary
            'depth': depth.astype(np.float32),
            'depth_neg': depth_neg.astype(np.float32),
            'name': name  # Sequence and zero-padded frame index, e.g., 02_000013
        }

        # Apply transformations if specified
        if self.transform:
            data_item = self.transform(data_item)

        return data_item


class DataGenerator(object):
    def __init__(self,
                 datadir,
                 phase,
                 perturb_filenames,
                 high_gpu=True,
                 augmentation=None,
                 loader='kitti_raw',
                 intrinsic=False):
        self.phase = phase
        self.high_gpu = high_gpu
        self.perturb_filenames = perturb_filenames
        self.augmentation = augmentation
        self.loader = loader
        self.intrinsic = intrinsic

        # if not self.phase in ['train', 'test', 'val', 'check', 'checkval']:
        #     raise ValueError("Panic::Invalid phase parameter")
        # else:
        #     pass

        transformer = CustTransformer(self.phase)

        if self.loader == 'kitti_raw':
            self.dataset = KittiDataset(datadir,
                                        self.phase,
                                        self.perturb_filenames,
                                        transformer.get_transform(),
                                        augmentation=self.augmentation)
        elif self.loader == 'kitti_odom':
            self.dataset = KITTIOdometryDataset(datadir=datadir, phase=self.phase, perturb_filenames=perturb_filenames,
                                                transform=transformer.get_transform(), augmentation=augmentation, intrinsic=self.intrinsic)
        # TODO: Implement self.loader == 'WOD' for Waymo Open Dataset
        
        else:
            raise NotImplementedError(f"Loader '{self.loader}' not implemented.")

    def create_data(self, batch_size, nthreads=0, shuffle=False):
        print(f'num_workers: {nthreads}')
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=shuffle,
                          num_workers=nthreads,
                          drop_last=True,  # Ensures all batches are the same size
                          pin_memory=self.high_gpu)


class KittiLeftImageDataset(KittiDataset):
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=None):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        left_img = data_item['left_img']  # Just extract the left image
        return left_img


class KittiDepthDataset(KittiDataset):
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=None):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        depth = data_item['depth']  # Just extract the depth
        return depth

class KittiNegDataset(KittiDataset):
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=None):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        depth_neg = data_item['depth_neg']  # Just extract the depth
        return depth_neg

class KittiOdomLRGB(KITTIOdometryDataset):
    def __init__(self, datadir, phase, perturb_filenames, cam_index=2, transform=None, augmentation=None):
        super().__init__(datadir, phase, perturb_filenames, cam_index=cam_index, transform=transform,
                         augmentation=augmentation)

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        left_img = data_item['left_img']
        return left_img


class KittiOdomDepth(KITTIOdometryDataset):
    def __init__(self, datadir, phase, perturb_filenames, cam_index=2, transform=None, augmentation=None):
        super().__init__(datadir, phase, perturb_filenames, cam_index=cam_index, transform=transform, augmentation=augmentation)

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        depth = data_item['depth']
        return depth


class KittiOdomDepthNeg(KITTIOdometryDataset):
    def __init__(self, datadir, phase, perturb_filenames, cam_index=2, transform=None, augmentation=None):
        super().__init__(datadir, phase, perturb_filenames, cam_index=cam_index, transform=transform, augmentation=augmentation)

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        depth_neg = data_item['depth_neg']
        return depth_neg

def create_dataloaders(root, perturb_filenames, mode, batch_size, num_cores, augmentation, loader='kitti_raw'):
    # Initialize the transformer based on mode
    transformer = CustTransformer(mode)
    transform = transformer.get_transform()

    # Dataset for left images with the transform
    if loader == 'kitti_raw':
        left_img_dataset = KittiLeftImageDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform, augmentation=augmentation)
        dataloader_img = DataLoader(left_img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                    drop_last=True, pin_memory=True)

        # Dataset for depth with the transform
        depth_dataset = KittiDepthDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform, augmentation=augmentation)
        dataloader_lid = DataLoader(depth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                    drop_last=True, pin_memory=True)

        # Dataset for depth with the transform
        depth_dataset_neg = KittiNegDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform, augmentation=augmentation)
        dataloader_neg = DataLoader(depth_dataset_neg, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                    drop_last=True, pin_memory=True)
    elif loader == 'kitti_odom':
        left_img_dataset = KittiOdomLRGB(root, mode, perturb_filenames=perturb_filenames, transform=transform, augmentation=augmentation)
        dataloader_img = DataLoader(left_img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                    drop_last=True, pin_memory=True)

        # Dataset for depth with the transform
        depth_dataset = KittiOdomDepth(root, mode, perturb_filenames=perturb_filenames, transform=transform, augmentation=augmentation)
        dataloader_lid = DataLoader(depth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                    drop_last=True, pin_memory=True)

        # Dataset for depth with the transform
        depth_dataset_neg = KittiOdomDepthNeg(root, mode, perturb_filenames=perturb_filenames, transform=transform, augmentation=augmentation)
        dataloader_neg = DataLoader(depth_dataset_neg, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                    drop_last=True, pin_memory=True)
    else:
        raise NotImplementedError(f"Loader '{loader}' not implemented.")

    print("data is loaded")
    return dataloader_img, dataloader_lid, dataloader_neg
