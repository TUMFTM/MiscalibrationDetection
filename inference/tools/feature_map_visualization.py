from torchvision.models import resnet18
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.calc_receptive_field import PixelwiseFeatureMaps
from src.datasets.dataloader.dataset_2D import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from src.utils.save_load_model import load_model_lidar, load_model_img

device = torch.device("cuda:0")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
kitti_path = os.path.join(root, 'data', 'kitti')

# Load pretrained model
# im_pretrained_path = os.path.join(root, 'outputs/models', "image_240719_full_1")
# lid_pretrained_path = os.path.join(root, 'outputs/models', "lidar_240719_full_1")
im_pretrained_path = os.path.join(root, 'outputs/models', "240725_full_1_image")
lid_pretrained_path = os.path.join(root, 'outputs/models', "240725_full_1_lid")
im_pretrained = load_model_img(im_pretrained_path)
lid_pretrained = load_model_lidar(lid_pretrained_path)

model_im = im_pretrained.to(device)
model_lid = lid_pretrained.to(device)

eval_gen = DataGenerator(kitti_path, 'check', perturb_filenames="perturbation_neg.csv")
eval_dataloader = eval_gen.create_data(8)

masking = False
pixel_wise = False

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
        distance_map = distance.unsqueeze(1)
        distance_map = PixelwiseFeatureMaps(model=model_im, embeddings_value=distance_map, input_image_size=(H, W))
        distance_map = distance_map.assign_embedding_value().squeeze(1)
        distance_map = torch.pow(distance_map, 2)

        if mask is not None and mask.any():
            loss_contrastive_nomask = torch.mean(distance_map)
            distance_map = distance_map * mask
            loss_contrastive = torch.sum(distance_map) / torch.count_nonzero(mask)
        else:
            loss_contrastive = loss_contrastive_nomask = torch.mean(distance_map)
    else:
        distance_map = torch.pow(distance, 2)
        loss_contrastive = loss_contrastive_nomask = torch.mean(distance_map)

    return loss_contrastive, loss_contrastive_nomask, distance_map

model_im.eval()
model_lid.eval()

with torch.no_grad():
    for batch in eval_dataloader:
        left_img_batch = batch['left_img'].to(device)
        depth_batch = batch['depth'].to(device)
        depth_neg = batch['depth_neg'].to(device)
        name = batch['name']

        i = 5

        image_sample = left_img_batch[i]

        # Positive Sample
        lid_pos_sample = depth_batch[i]

        # Normal Negative Sample (pertubated)
        # lid_neg_sample = depth_neg[i]

        # Patched
        # lid_neg_sample = depth_batch[i].clone()
        # lid_neg_sample[:, 75:125,75:250] = 0

        # Scaled
        # lid_neg_sample = depth_batch[i].clone()
        # lid_neg_sample[:, 75:125,75:250] = 0.2 * lid_neg_sample[:, 75:125,75:250]

        # Random Assigned
        lid_neg_sample = depth_batch[i+2]
        # print(f"file name: {name[i]}")

        image_lidar_visualization(image=image_sample, lid_pos=lid_pos_sample, lid_neg=lid_neg_sample)

        # Prediction & Backpropagation
        pred_im = model_im.forward(image_sample.unsqueeze(0))
        pred_lid = model_lid.forward(lid_pos_sample.unsqueeze(0))
        pred_neg = model_lid.forward(lid_neg_sample.unsqueeze(0))

        N, C, H, W = left_img_batch.size()

        if masking:
            mask = (lid_pos_sample > 0.0).int()
            mask_neg = (lid_neg_sample > 0.0).int()
            # print(mask.shape)
        else:
            mask = None
            mask_neg = None

        cl_loss, cl_loss_nomask, loss = loss_map(output_im=pred_im, output_lid=pred_lid, model_im=model_im, H=H, W=W,
                        pixel_wise=pixel_wise, mask=mask)
        print(f"cl_loss:{cl_loss}")
        print(f"cl_nomask:{cl_loss_nomask}")

        cl_neg, cl_nomask_neg, loss_neg = loss_map(output_im=pred_im, output_lid=pred_neg, model_im=model_im, H=H, W=W,
                            pixel_wise=pixel_wise, mask=mask_neg)
        print(f"cl_neg: {cl_neg}")
        print(f"cl_neg_nomask:{cl_nomask_neg}")

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

        # for i in range(channel_embeddings):
        #     torch.set_printoptions(profile="full")
        #     feature_im1 = pred_im[0,i,:,:]
        #     feature_lid1 = pred_lid[0,i,:,:]
        #     feature_lid1_neg = pred_neg[0,i,:,:]
            
        #     feature_im1_np = feature_im1.cpu().numpy()
        #     feature_lid1_np = feature_lid1.cpu().numpy()
        #     feature_lid1_np_neg = feature_lid1_neg.cpu().numpy()

            
        #     print(f'image: {feature_im1}')
        #     print(f'lidar pos: {feature_lid1}')
        #     print(f'lidar neg: {feature_lid1_neg}')

        #     max_value = (max(max(np.max(feature_lid1_np), np.max(feature_im1_np)), np.max(feature_lid1_np_neg)))

        #     plt.figure(figsize=(15, 6))  
        #     plt.imshow(feature_im1_np, cmap='viridis', vmin=0, vmax=max_value)  
        #     plt.axis('off') 
        #     plt.text(0.5, -0.1, f'image, max:{max_value:.3f}, layer:{i+1}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
        #     plt.tight_layout()  
        #     plt.show()
            
        #     plt.figure(figsize=(15, 6))  
        #     plt.imshow(feature_lid1_np, cmap='viridis', vmin=0, vmax=max_value)  
        #     plt.axis('off') 
        #     plt.text(0.5, -0.1, f'lidar pos, max:{max_value:.3f}, layer:{i+1}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
        #     plt.tight_layout()  
        #     plt.show()

        #     plt.figure(figsize=(15, 6))  
        #     plt.imshow(feature_lid1_np_neg, cmap='viridis', vmin=0, vmax=max_value)  
        #     plt.axis('off') 
        #     plt.text(0.5, -0.1, f'lidar neg, max:{max_value:.3f}, layer:{i+1}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
        #     plt.tight_layout()  
        #     plt.show()