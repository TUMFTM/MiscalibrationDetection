import torch
import torch.nn as nn
from src.utils.calc_receptive_field import PixelwiseFeatureMaps
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0, mode='resnet'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.mode = mode


    def forward(self, output_im, output_lid, labels, model_im, H, W, vit, mask):
        # L2 Distances of feature embeddings

        if self.mode == 'vit':
            # print("Using CL for ViT")
            # Input (B, N, D). Comparing each embedding (1, D)-vectors
            distances = F.pairwise_distance(output_im, output_lid, p=2)  # Shape (B, N)
            distances_mean = distances.mean(dim=1)  # Shape (B,)

            positive_loss = torch.pow(distances_mean, 2) * labels
            negative_loss = torch.pow(torch.clamp(self.margin - distances_mean, min=0.0), 2) * (1 - labels)

            loss_contrastive = (positive_loss + negative_loss).mean()

        elif self.mode == 'resnet':
            # print("Using pixel-wise CL for ResNet")
            # Input (B, C, H, W). Comparing each pixel (across C)
            distance = torch.sqrt(torch.sum((output_im - output_lid) ** 2, dim=1))
            N, H_dist, W_dist = distance.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)

            positive_loss = torch.pow(distance, 2) * labels_broadcasted
            negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)
            loss_contrastive = (positive_loss + negative_loss).mean()

        elif self.mode == 'resnet_instance':
            N = output_im.shape[0]

            # Calculate the Euclidean distance between the pooled outputs
            euclidean_distance = F.pairwise_distance(output_im, output_lid)  # Shape: [N]

            # Ensure labels are shaped correctly for batch-wise contrastive loss
            labels = labels.view(N)  # Shape: [N]

            # Compute the contrastive loss
            loss_contrastive = (1 - labels) * torch.pow(euclidean_distance, 2) + \
                               labels * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

            # Take the mean loss over the batch
            loss_contrastive = loss_contrastive.mean()
        else:
            raise ValueError(f"Mode '{self.mode}' is not supported. Choose from 'vit', 'resnet', or 'resnet_instance'.")

        return loss_contrastive
