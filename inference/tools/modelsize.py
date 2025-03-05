import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import yaml
from src.models.classifier_head import classifier_head
from src.utils.helper import load_checkpoint, load_checkpoint_cls, init_model, init_opt


parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_name', type=str,
    help='name of lidar model to save',
    default='lidar_backbone')


if __name__ == "__main__":
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    kitti_path = os.path.join(root, 'data', 'kitti')
    config_name = 'configs_' + args.save_name + '.yaml'
    configs_path = os.path.join(root, 'configs', config_name)

    with open(configs_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('cuda is not available')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    path_encoders = os.path.join(root, 'outputs_gpu', args.save_name, 'models', f'{args.save_name}_contrastive-latest.pth.tar')
    path_cls = os.path.join(root, 'outputs_gpu', args.save_name, 'models', f'{args.save_name}_classifier-latest.pth.tar')

    # Load the checkpoint
    ckpt_enc = torch.load(path_encoders)
    ckpt_cls = torch.load(path_cls)

    # Extract the model's state dict (the parameters)
    model_state_dict_im = ckpt_enc['encoder_im']
    model_state_dict_lid = ckpt_enc['encoder_lid']
    model_state_dict_cls = ckpt_cls['classifier']

    # Calculate the size of the model parameters (image encoder)
    model_size_im = sum(param.numel() * param.element_size() for param in model_state_dict_im.values())

    # Convert the size from bytes to MB
    model_size_mb_im = model_size_im / (1024 * 1024)
    print(f"Model size image encoders: {model_size_mb_im:.2f} MB")

    # Convert the size to millions of parameters
    model_size_m_im = sum(param.numel() for param in model_state_dict_im.values()) / 1e6
    print(f"Model size image encoders: {model_size_m_im:.3f} M parameters")

    # Calculate the size of the model parameters (LiDAR encoder)
    model_size_lid = sum(param.numel() * param.element_size() for param in model_state_dict_lid.values())

    # Convert the size from bytes to MB
    model_size_mb = model_size_lid / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")

    # Convert the size to millions of parameters
    model_size_m_lid = sum(param.numel() for param in model_state_dict_lid.values()) / 1e6
    print(f"Model size LiDAR encoders: {model_size_m_lid:.3f} M parameters")

    # Calculate the size of the classifier model
    model_size = sum(param.numel() * param.element_size() for param in model_state_dict_cls.values())

    # Convert the size from bytes to MB
    model_size_mb = model_size / (1024 * 1024)
    print(f"Model size classifier: {model_size_mb:.2f} MB")

    # Convert the size to millions of parameters
    model_size_m = sum(param.numel() for param in model_state_dict_cls.values()) / 1e6
    print(f"Model size classifier: {model_size_m:.3f} M parameters")
