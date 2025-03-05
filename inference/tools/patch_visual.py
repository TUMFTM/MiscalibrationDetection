import torch
import torch.nn.functional as F

def patch_wise_analysis(lidar_mask_downsampled, patch_size=4):
    N, H_dist, W_dist = lidar_mask_downsampled.shape
    
    pad_h = (patch_size - H_dist % patch_size) % patch_size
    pad_w = (patch_size - W_dist % patch_size) % patch_size
    
    lidar_mask_padded = F.pad(lidar_mask_downsampled, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    lidar_mask_reshaped = lidar_mask_padded.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    print(lidar_mask_reshaped.shape)
    
    patch_has_nonzero = lidar_mask_reshaped.sum(dim=(-1, -2)) > 0
    
    patch_result = patch_has_nonzero.float().repeat_interleave(patch_size, dim=-1).repeat_interleave(patch_size, dim=-2)
    
    lidar_mask_analyzed_padded = patch_result[:, :H_dist + pad_h, :W_dist + pad_w]
    
    lidar_mask_analyzed = lidar_mask_analyzed_padded[:, :H_dist, :W_dist]
    
    return lidar_mask_analyzed

lidar_mask_downsampled = torch.rand(64, 22, 72)
lidar_mask_analyzed = patch_wise_analysis(lidar_mask_downsampled, patch_size=4)

print(lidar_mask_analyzed.shape)