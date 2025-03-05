import os.path
import torch
import yaml
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from cka_analysis import cka_analysis
from src.utils.helper import full_load_latest


parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_name_1', type=str,
    help='model name',
    default='resnet18_small_aug_240917')
parser.add_argument(
    '--save_name_2', type=str,
    help='model name',
    default='resnet18_small_aug_240917')
parser.add_argument(
    '--perturbation', type=str,
    help='type of evaluation',
    default='neg_master')


# Cross CKA Analysis
if __name__ == "__main__":
    args = parser.parse_args()

    # Directory
    save_name_1 = args.save_name_1
    save_name_2 = args.save_name_2
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    kitti_path = os.path.join(root, 'data', 'kitti')
    config_name_1 = 'configs_' + save_name_1 + '.yaml'
    config_name_2 = 'configs_' + save_name_2 + '.yaml'
    configs_path_1 = os.path.join(root, 'configs', config_name_1)
    configs_path_2 = os.path.join(root, 'configs', config_name_2)
    perturbation_file = 'perturbation_' + args.perturbation + '.csv'
    # -

    # Configs loader, CUDA
    with open(configs_path_1, 'r') as y_file:
        params_1 = yaml.load(y_file, Loader=yaml.FullLoader)
    with open(configs_path_2, 'r') as y_file:
        params_2 = yaml.load(y_file, Loader=yaml.FullLoader)
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('cuda is not available')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    # -

    # Saving directories
    output_dir = os.path.join(root, 'inference/eval', f"{save_name_1}_{save_name_2}_{args.perturbation}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")
    else:
        raise FileNotFoundError(f"Failed to create directory {output_dir}.")

    if os.path.exists(output_dir) and not os.listdir(output_dir):
        save_dir = os.path.join(output_dir, 'run_1')
        print(f"Directory {save_dir} created.")
    else:
        num_folders = len([name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))])
        save_dir = os.path.join(output_dir, f'run_{num_folders + 1}')
        print(f"Directory {save_dir} created.")
    # -

    encoder_im_1, encoder_lid_1, classifier_1 = full_load_latest(device=device,
                                                                 params=params_1,
                                                                 root=root,
                                                                 save_name=save_name_1)

    encoder_im_2, encoder_lid_2, classifier_2 = full_load_latest(device=device,
                                                                 params=params_2,
                                                                 root=root,
                                                                 save_name=save_name_2)

    if '_aug' in save_name_1:
        tag_1 = "Calib (II)"
    elif '_noaug' in save_name_1:
        tag_1 = "Calib (I)"
    else:
        assert False, "Error: save_path must contain '_aug' or '_noaug'"

    if '_aug' in save_name_2:
        tag_2 = "Calib (II)"
    elif '_noaug' in save_name_2:
        tag_2 = "Calib (I)"
    else:
        assert False, "Error: save_path must contain '_aug' or '_noaug'"

    print("starting cka analysis")
    cka_analysis(data_root=kitti_path, output_dir=save_dir, model_1=encoder_lid_1, model_2=encoder_lid_2,
                 tag_1=f'{tag_1} LiDAR layers', tag_2=f'{tag_2} LiDAR layers', title="LiDAR Encoders",
                 perturbation_eval=perturbation_file, show_plot=False, crossmodel=True)
    cka_analysis(data_root=kitti_path, output_dir=save_dir, model_1=encoder_im_1, model_2=encoder_im_2,
                 tag_1=f'{tag_1} Image layers', tag_2=f'{tag_2} Image layers', title="Image Encoders",
                 perturbation_eval=perturbation_file, show_plot=False, crossmodel=True)
