import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import torch.multiprocessing as mp
from train_resnet import main as train_resnet
import yaml
import torch
from src.models.classifier_head import classifier_head
from src.utils.helper import load_checkpoint, load_checkpoint_cls, init_model, init_opt


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs_resnet18_small_base.yaml')
parser.add_argument(
    '--save_name', type=str,
    help='name of model for saving',
    default='dates')
parser.add_argument(
    '--vit', action='store_true', help='comparing pixel-wise distance')
parser.add_argument(
    '--masking', action='store_true', help='enable masking')
parser.add_argument(
    '--classifier', action='store_true', help='training directly classifier')


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)

    args = parser.parse_args()
    save_name = args.save_name

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_name = 'configs_' + save_name + '.yaml'
    configs_path = os.path.join(root, 'configs', config_name)
    kitti_path = os.path.join(root, 'data', 'kitti')

    # Load configs
    with open(configs_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    # Printing logs:
    if args.classifier:
        print('Training the classifier')
    else:
        print('Not training the classifier')
    curr_model_name = params['meta']['model_name']
    print(f'Training model with {curr_model_name}, model name: {args.save_name}')

    # Train Model
    mode = params['meta']['backbone']
    if mode in ['resnet', 'vit']:
        train_resnet(args=params, project_root=root, save_name=save_name, vit=args.vit, masking=args.masking, logger_launch='True',
                     train_classifier=args.classifier)
    else:
        assert mode in ['resnet', 'vit'], 'backbone is not covered'

    # Loading Model testing:
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('cuda is not available')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    tag_encoders = params['logging']['tag']
    tag_cls = params['logging_cls']['tag']
    path_encoders = os.path.join(root, 'outputs/models', f'{args.save_name}_{tag_encoders}-latest.pth.tar')
    path_cls = os.path.join(root, 'outputs/models', f'{args.save_name}_{tag_cls}-latest.pth.tar')

    encoder_im, encoder_lid = init_model(device=device, mode=params['meta']['backbone'], model_name=params['meta']['model_name'])
    encoder_im.eval()
    encoder_lid.eval()
    opt_im, scheduler_im = init_opt(encoder_im, params['optimization'])
    opt_lid, scheduler_lid = init_opt(encoder_lid, params['optimization'])
    encoder_im, encoder_lid, opt_im, opt_lid, epoch = load_checkpoint(r_path=path_encoders,
                                                                      encoder_im=encoder_im,
                                                                      encoder_lid=encoder_lid,
                                                                      opt_im=opt_im, opt_lid=opt_lid)

    classifier = classifier_head(model_im=encoder_im, model_lid=encoder_lid, model_name=params['meta']['model_name'])
    classifier, epoch_cls = load_checkpoint_cls(r_path=path_cls, classifier=classifier)
