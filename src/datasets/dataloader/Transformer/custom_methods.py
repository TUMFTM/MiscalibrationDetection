import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as Fun
from PIL import Image
from .base_methods import BaseMethod
import torchvision.transforms.functional as TF


"""
This file defines some example transforms.
Each transform method is defined by using BaseMethod class
"""


class TransToPIL(BaseMethod):
    """
    Transform method to convert images as PIL Image.
    """
    def __init__(self):
        BaseMethod.__init__(self)
        self.to_pil = transforms.ToPILImage()

    def __call__(self, data_item):
        self.set_data(data_item)

        if not self._is_pil_image(self.left_img):
            data_item['left_img'] = self.to_pil(self.left_img)
        if not self._is_pil_image(self.depth):
            data_item['depth'] = Image.fromarray(self.depth)
        if not self._is_pil_image(self.depth_neg):
            data_item['depth_neg'] = Image.fromarray(self.depth_neg)

        return data_item


class Scale(BaseMethod):
    def __init__(self, mode, size):
        BaseMethod.__init__(self, mode)
        self.scale = transforms.Resize(size, Image.BILINEAR)
        self.size = size

    def _downscale_lidar_tensor(self, lidar_tensor):
        # N, original_height, original_width = lidar_tensor.shape
        new_height = self.size[0]
        new_width = self.size[1]
        # Create mask for non-zero values
        mask = lidar_tensor != 0

        # Interpolate the non-zero values
        non_zero_values = lidar_tensor.float() * mask.float()
        non_zero_interpolated = Fun.interpolate(non_zero_values.unsqueeze(1), size=(new_height, new_width),
                                              mode='bilinear', align_corners=False)
        mask_interpolated = Fun.interpolate(mask.float().unsqueeze(1), size=(new_height, new_width), mode='bilinear',
                                          align_corners=False)

        # Avoid division by zero
        mask_interpolated[mask_interpolated == 0] = 1

        # Combine interpolated values with the original zeros
        lidar_tensor_downscaled = (non_zero_interpolated / mask_interpolated).squeeze(1)

        # Restore zero values
        lidar_tensor_downscaled[mask_interpolated.squeeze(1) == 0] = 0

        return lidar_tensor_downscaled

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode in ["pair", "Img"]:
            data_item['left_img'] = self.scale(self.left_img)
        if self.mode in ["pair", "depth"]:
            data_item['depth'] = self._downscale_lidar_tensor(self.depth)
            data_item['depth_neg'] = self._downscale_lidar_tensor(self.depth_neg)

        return data_item


class RandomHorizontalFlip(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            data_item['left_img'] = self.left_img.transpose(Image.FLIP_LEFT_RIGHT)
            data_item['depth'] = self.depth.transpose(Image.FLIP_LEFT_RIGHT)
            data_item['depth_neg'] = self.depth_neg.transpose(Image.FLIP_LEFT_RIGHT)

        return data_item


class RandomRotate(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def rotate_pil_func():
        degree = random.randrange(-500, 500)/100
        return (lambda pil, interp : F.rotate(pil, degree, interp))

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            rotate_pil = self.rotate_pil_func()
            data_item['left_img'] = rotate_pil(self.left_img, Image.BICUBIC)
            data_item['depth'] = rotate_pil(self.depth, Image.BILINEAR)
            data_item['depth_neg'] = rotate_pil(self.depth_neg, Image.BILINEAR)

        return data_item


class ImgAug(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def adjust_pil(pil):
        brightness = random.uniform(0.8, 1.0)
        contrast = random.uniform(0.8, 1.0)
        saturation = random.uniform(0.8, 1.0)

        pil = F.adjust_brightness(pil, brightness)
        pil = F.adjust_contrast(pil, contrast)
        pil = F.adjust_saturation(pil, saturation)

        return pil

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['left_img'] = self.adjust_pil(self.left_img)

        return data_item


class ToTensor(BaseMethod):
    def __init__(self, mode):
        BaseMethod.__init__(self, mode=mode)
        self.totensor = transforms.ToTensor()

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode == "Img":
            data_item['left_img'] = self.totensor(self.left_img)

        if self.mode == "depth":
            data_item['depth'] = self.totensor(self.depth)
            data_item['depth_neg'] = self.totensor(self.depth_neg)

        return data_item


class ImgNormalize(BaseMethod):
    def __init__(self, mean, std):
        BaseMethod.__init__(self)
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['left_img'] = self.normalize(self.left_img)

        return data_item


def normalize_lidar(lidar):
    mean_lidar = lidar.mean()
    std_lidar = lidar.std()
    # Normalize the LiDAR data
    normalized_lidar = (lidar - mean_lidar) / std_lidar

    return normalized_lidar


class InputNormalize(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)
        self.normalize_im = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std from Imagenet

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['left_img'] = self.normalize_im(self.left_img)
        data_item['depth'] = normalize_lidar(self.depth)
        data_item['depth_neg'] = normalize_lidar(self.depth_neg)

        return data_item


class RandCrop(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def _random_crop(self, img, depth, depth_neg):
        combined = torch.cat((img, depth, depth_neg), dim=0)

        transform = transforms.RandomCrop((176, 576))
        # transform = transforms.RandomCrop((176, 176))
        cropped = transform(combined)

        img_cropped = cropped[:3, :, :]
        lid_cropped = cropped[3, :, :]
        neg_cropped = cropped[4,:,:]

        lid_cropped = lid_cropped.unsqueeze(0)
        neg_cropped = neg_cropped.unsqueeze(0)

        return img_cropped, lid_cropped, neg_cropped

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['left_img'], data_item['depth'], data_item['depth_neg'] = self._random_crop(self.left_img, self.depth, self.depth_neg)

        return data_item


class TransToTensor(BaseMethod):
    """
    Convert data items to torch tensors and move them to GPU.
    """
    def __call__(self, data_item):
        self.set_data(data_item)

        # Convert 'left_img' to tensor and move to GPU
        if isinstance(self.left_img, Image.Image):
            data_item['left_img'] = TF.to_tensor(self.left_img)
        elif isinstance(self.left_img, np.ndarray):
            data_item['left_img'] = torch.from_numpy(self.left_img).permute(2, 0, 1)

        # Convert 'depth' to tensor and move to GPU
        if isinstance(self.depth, np.ndarray):
            data_item['depth'] = torch.from_numpy(self.depth).unsqueeze(0)

        # Convert 'depth_neg' to tensor and move to GPU
        if isinstance(self.depth_neg, np.ndarray):
            data_item['depth_neg'] = torch.from_numpy(self.depth_neg).unsqueeze(0)

        return data_item


class ScaleGPU(BaseMethod):
    def __init__(self, size):
        BaseMethod.__init__(self)
        self.size = size  # Target size as (new_height, new_width)

    def _downscale_lidar_tensor(self, lidar_tensor):
        """
        Downscale the LiDAR tensor, ensuring it has 4 dimensions (N, C, H, W) for interpolation.
        """
        # If lidar_tensor is 2D, add channel and batch dimensions

        # New target height and width
        new_height, new_width = self.size

        # Create mask for non-zero values
        mask = lidar_tensor != 0

        # Interpolate the non-zero values
        non_zero_values = lidar_tensor.float() * mask.float()
        non_zero_interpolated = Fun.interpolate(non_zero_values.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)
        mask_interpolated = Fun.interpolate(mask.float().unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Avoid division by zero
        mask_interpolated[mask_interpolated == 0] = 1

        # Combine interpolated values with the original zero values
        lidar_tensor_downscaled = (non_zero_interpolated / mask_interpolated)

        # Remove batch and channel dimensions
        lidar_tensor_downscaled = lidar_tensor_downscaled.squeeze(0)

        # Restore zero values in the tensor
        lidar_tensor_downscaled[mask_interpolated.squeeze(0) == 0] = 0
        return lidar_tensor_downscaled

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['left_img'] = Fun.interpolate(data_item['left_img'].unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        data_item['depth'] = self._downscale_lidar_tensor(data_item['depth'])
        data_item['depth_neg'] = self._downscale_lidar_tensor(data_item['depth_neg'])

        return data_item


class RandomHorizontalFlipGPU(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def __call__(self, data_item):
        self.set_data(data_item)
        if random.random() < 0.5:
            data_item['left_img'] = torch.flip(data_item['left_img'], [-1])  # Flip horizontally
            data_item['depth'] = torch.flip(data_item['depth'], [-1])
            data_item['depth_neg'] = torch.flip(data_item['depth_neg'], [-1])
        return data_item


class RandomRotateGPU(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def __call__(self, data_item):
        self.set_data(data_item)
        degree = random.uniform(-30, 30)
        data_item['left_img'] = F.rotate(data_item['left_img'], degree)
        data_item['depth'] = F.rotate(data_item['depth'], degree)
        data_item['depth_neg'] = F.rotate(data_item['depth_neg'], degree)
        return data_item


class ImgAugGPU(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def __call__(self, data_item):
        self.set_data(data_item)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        data_item['left_img'] = F.adjust_brightness(data_item['left_img'], brightness)
        data_item['left_img'] = F.adjust_contrast(data_item['left_img'], contrast)
        data_item['left_img'] = F.adjust_saturation(data_item['left_img'], saturation)
        return data_item


class RandCropGPU(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def _random_crop(self, img, depth, depth_neg):
        combined = torch.cat((img, depth, depth_neg), dim=0)
        cropped = F.center_crop(combined, (176, 576))
        img_cropped = cropped[:3, :, :]
        lid_cropped = cropped[3, :, :].unsqueeze(0)
        neg_cropped = cropped[4, :, :].unsqueeze(0)
        return img_cropped, lid_cropped, neg_cropped

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item['left_img'], data_item['depth'], data_item['depth_neg'] = self._random_crop(data_item['left_img'],
                                                                                              data_item['depth'],
                                                                                              data_item['depth_neg'])
        return data_item

