from torchvision.models import resnet18
import torch.nn.functional as F
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.calc_receptive_field import PixelwiseFeatureMaps
from src.datasets.dataloader.dataset_2D import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from src.utils.save_load_model import load_model_lidar, load_model_img
from src.utils.helper import load_checkpoint
from src.models.mm_siamese import resnet18_2B_lid, resnet18_2B_im
from src.models.classifier_head import classifier_head


def lidar_scatter(lidar):
    lidar = lidar.permute(1, 2, 0).cpu().numpy()
    values_store = []
    lidar = np.squeeze(lidar)

    for (i, j), value in np.ndenumerate(lidar):
        values_store.append([j, i, value])

    values_store = np.array(values_store)
    values_store = np.delete(values_store, np.where(values_store[:, 2] == 0), axis=0)

    return values_store

def image_lidar_visualization(image, lid_pos, lid_neg):

    mask = (lid_pos != 0).int()
    # Permute
    img_np1 = image.permute(1, 2, 0).cpu().numpy()
    values_store = lidar_scatter(lid_pos)
    values_store_neg = lidar_scatter(lid_neg)
    values_store_mask = lidar_scatter(mask)

    check = (values_store[:,0] == values_store_mask[:,0])

    plt.figure(figsize=(15, 4.8))
    plt.imshow(img_np1, alpha=1.0)
    plt.scatter(values_store[:, 0], values_store[:, 1], c=values_store[:, 2], cmap='rainbow_r', alpha=0.5, s=3)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 4.8))
    plt.imshow(img_np1)
    plt.scatter(values_store_neg[:, 0], values_store_neg[:, 1], c=values_store_neg[:, 2], cmap='rainbow_r', alpha=0.5,
                s=3)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 4.8))
    plt.imshow(img_np1, alpha=0.0)
    plt.scatter(values_store_mask[:, 0], values_store_mask[:, 1], c=values_store_mask[:, 2], cmap='grey', alpha=0.5,
                s=5)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def loss_map(output_im, output_lid, model_im, H, W, pixel_wise, mask):
    # L2 Distances of feature embeddings
    dist_squared = (output_im - output_lid) ** 2
    summed = torch.sum(dist_squared, dim=1)
    distance = torch.sqrt(summed)

    if pixel_wise:
        distance = PixelwiseFeatureMaps(model=model_im, embeddings_value=distance.unsqueeze(1),
                                        input_image_size=(H, W))
        distance = distance.assign_embedding_value().squeeze(1)
        N, H_dist, W_dist = distance.shape

        mask = mask.float()

        patch_size = 16

        # Reshape the mask into patches of size 4x4
        mask_reshaped = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        # Check if each patch has at least one non-zero value
        patch_has_nonzero = mask_reshaped.sum(dim=(-1, -2)) > 0

        # Expand the result back to the original size
        patch_result = patch_has_nonzero.float().repeat_interleave(patch_size, dim=-1).repeat_interleave(patch_size,
                                                                                                         dim=-2)

        # Reshape back to the original mask size
        mask_analyzed = patch_result.view(N, 1, H, W)

        distance_final = distance * mask_analyzed
        distance_map = torch.pow(distance_final, 2)
        loss_contrastive_nomask = (torch.pow(distance, 2)).mean()
        loss_contrastive = distance_map.mean()
    else:
        distance_map = torch.pow(distance, 2)
        loss_contrastive = loss_contrastive_nomask = torch.mean(distance_map)

    return loss_contrastive, loss_contrastive_nomask, distance_map

def patched_mask_visual(image, mask, patch_size=4):
    mask = mask.unsqueeze(0)
    N, C, H, W = mask.shape
    
    # Reshape the mask into patches of size 4x4
    mask_reshaped = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Check if each patch has at least one non-zero value
    patch_has_nonzero = mask_reshaped.sum(dim=(-1, -2)) > 0
    
    # Expand the result back to the original size
    patch_result = patch_has_nonzero.float().repeat_interleave(patch_size, dim=-1).repeat_interleave(patch_size, dim=-2)
    
    # Reshape back to the original mask size
    lidar_mask_analyzed = patch_result.view(N, C, H, W).squeeze(0)

    values_store_mask = lidar_scatter(lidar_mask_analyzed)
    img_np1 = image.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(15, 4.8))
    plt.imshow(img_np1, alpha=0.0)
    plt.scatter(values_store_mask[:, 0], values_store_mask[:, 1], c=values_store_mask[:, 2], cmap='grey', alpha=0.5,
                s=5)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    
    return lidar_mask_analyzed

if __name__ == "__main__":
    device = torch.device("cuda:0")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    kitti_path = os.path.join(root, 'data', 'kitti')

    # Load pretrained model
    save_name = '240813'
    path = os.path.join(root, 'outputs_gpu', 'outputs/models', f'{save_name}_contrastive-latest.pth.tar')

    im_pretrained, lid_pretrained, cls_pretrained, epoch = load_checkpoint(r_path=path, encoder_im=resnet18_2B_im(), encoder_lid=resnet18_2B_lid(),
                                                                           model_cls=classifier_head(model_im=resnet18_2B_im(), model_lid=resnet18_2B_lid()))

    model_im = im_pretrained.to(device)
    model_lid = lid_pretrained.to(device)

    eval_gen = DataGenerator(kitti_path, 'check', perturb_filenames="perturbation_neg_master.csv")
    eval_dataloader = eval_gen.create_data(8)

    masking = True
    pixel_wise = True
    mode_neg = True
    i = 5

    model_im.eval()
    model_lid.eval()

    with torch.no_grad():
        for batch in eval_dataloader:
            left_img_batch = batch['left_img'].to(device)
            depth_batch = batch['depth'].to(device)
            depth_neg = batch['depth_neg'].to(device)
            name = batch['name']

            image_sample = left_img_batch[i]
            lid_pos_sample = depth_batch[i]

            if mode_neg:
                # Normal Negative Sample (pertubated)
                lid_neg_sample = depth_neg[i]
            else:
                # Patched
                lid_neg_sample = depth_batch[i].clone()
                lid_neg_sample[:, 75:125,75:250] = 0

                # Scaled
                # lid_neg_sample = depth_batch[i].clone()
                # lid_neg_sample[:, 75:125,75:250] = 0.2 * lid_neg_sample[:, 75:125,75:250]

                # Random Assigned
                # lid_neg_sample = depth_batch[i+2]
                # print(f"file name: {name[i]}")

            if masking:
                mask = (lid_pos_sample != 0.0).int()
                mask_neg = (lid_neg_sample != 0.0).int()
                # print(mask.shape)
            else:
                mask = None
                mask_neg = None

            # Prediction
            pred_im = model_im.forward(image_sample.unsqueeze(0))
            pred_lid = model_lid.forward(lid_pos_sample.unsqueeze(0))
            pred_neg = model_lid.forward(lid_neg_sample.unsqueeze(0))
            N, C, H, W = left_img_batch.size()

            cl_loss, cl_loss_nomask, loss = loss_map(output_im=pred_im, output_lid=pred_lid, model_im=model_im, H=H, W=W,
                            pixel_wise=pixel_wise, mask=mask)
            print(f"cl_loss:{cl_loss}")
            print(f"cl_nomask:{cl_loss_nomask}")

            cl_neg, cl_nomask_neg, loss_neg = loss_map(output_im=pred_im, output_lid=pred_neg, model_im=model_im, H=H, W=W,
                                pixel_wise=pixel_wise, mask=mask_neg)
            print(f"cl_neg: {cl_neg}")
            print(f"cl_neg_nomask:{cl_nomask_neg}")

            # Visualization

            patched_mask_visual(image=image_sample, mask=mask)
            image_lidar_visualization(image=image_sample, lid_pos=lid_pos_sample, lid_neg=lid_neg_sample)

            # Convert the tensor to a NumPy array
            array = loss.cpu().numpy().squeeze(0)
            array_neg = loss_neg.cpu().numpy().squeeze(0)
            max_value = int(np.max(array_neg))

            text = f"CL loss: {cl_loss:.3f}"
            text_neg = f"CL loss: {cl_neg:.3f}"
            # Plot the array
            plt.figure(figsize=(15, 6))
            plt.imshow(array, cmap='viridis', vmin=0, vmax=max_value)
            plt.axis('off')
            plt.text(0.5, -0.1, text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(15, 6))
            plt.imshow(array_neg, cmap='viridis', vmin=0, vmax=max_value)
            plt.axis('off')
            plt.text(0.5, -0.1, text_neg, ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
            plt.tight_layout()
            plt.show()


            # Analyzing the embedding layers accross channel
            channel_embeddings = pred_im.size()[1]

            for i in range(channel_embeddings):
                torch.set_printoptions(profile="full")
                feature_im1 = pred_im[0,i,:,:]
                feature_lid1 = pred_lid[0,i,:,:]
                feature_lid1_neg = pred_neg[0,i,:,:]

                feature_im1_np = feature_im1.cpu().numpy()
                feature_lid1_np = feature_lid1.cpu().numpy()
                feature_lid1_np_neg = feature_lid1_neg.cpu().numpy()


                print(f'image: {feature_im1}')
                print(f'lidar pos: {feature_lid1}')
                print(f'lidar neg: {feature_lid1_neg}')

                max_value = (max(max(np.max(feature_lid1_np), np.max(feature_im1_np)), np.max(feature_lid1_np_neg)))

                plt.figure(figsize=(15, 6))
                plt.imshow(feature_im1_np, cmap='viridis', vmin=0, vmax=max_value)
                plt.axis('off')
                plt.text(0.5, -0.1, f'image, max:{max_value:.3f}, layer:{i+1}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(15, 6))
                plt.imshow(feature_lid1_np, cmap='viridis', vmin=0, vmax=max_value)
                plt.axis('off')
                plt.text(0.5, -0.1, f'lidar pos, max:{max_value:.3f}, layer:{i+1}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(15, 6))
                plt.imshow(feature_lid1_np_neg, cmap='viridis', vmin=0, vmax=max_value)
                plt.axis('off')
                plt.text(0.5, -0.1, f'lidar neg, max:{max_value:.3f}, layer:{i+1}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
                plt.tight_layout()
                plt.show()