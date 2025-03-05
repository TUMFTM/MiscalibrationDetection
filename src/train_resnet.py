import torch
from tqdm import tqdm
import os
from src.models.classifier_head import classifier_head
from src.datasets.dataloader.dataset_2D import DataGenerator
from src.utils.contrastive_loss import ContrastiveLoss as CL
from src.utils.helper import gen_mixed_data, init_model, load_checkpoint, init_opt
from src.utils.logger import tb_logger
import torch.nn as nn
from tensorboard import program
import multiprocessing


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def main(args, project_root, save_name, vit, masking, logger_launch='True', train_classifier='True'):

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('cuda is not available')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # LOGGING
    logger = tb_logger(args['logging'], project_root, save_name)
    if logger_launch:
        port = 6006
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', args['logging']['rel_path'], '--port', str(port)])
        url = tb.launch()
        print(f"TensorBoard started at {url}")
    checkpoint_freq = 4
    tag = args['logging']['tag']
    tag_cls = args['logging_cls']['tag']
    save_path = os.path.join(project_root, 'outputs/models', f'{save_name}_{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(project_root, 'outputs/models', f'{save_name}_{tag}-latest.pth.tar')
    # --

    # DATA
    batch_size = args['data']['batch_size']
    dataset_path = args['data']['dataset_path']
    perturbation_file = args['data']['perturbation_file']
    augmentation = args['data']['augmentation']
    loader = args['data']['loader']
    # --
    # Get the number of available CPU cores
    num_cores = min(multiprocessing.cpu_count(), 64)
    data_root = os.path.join(project_root, dataset_path)
    train_gen = DataGenerator(data_root, 'train', perturb_filenames=perturbation_file,
                              augmentation=augmentation, loader=loader)
    val_gen = DataGenerator(data_root, 'val', perturb_filenames=perturbation_file,
                            augmentation=augmentation, loader=loader)
    # train_gen = DataGenerator(data_root, 'debug', perturb_filenames=perturbation_file,
    #                           augmentation=augmentation, loader=loader)
    # val_gen = DataGenerator(data_root, 'debug', perturb_filenames=perturbation_file,
    #                         augmentation=augmentation, loader=loader)
    train_loader = train_gen.create_data(batch_size=batch_size, shuffle=True, nthreads=num_cores)
    val_loader = val_gen.create_data(batch_size=batch_size, shuffle=False, nthreads=num_cores)
    # --

    # MODEL
    pretrained_encoder = args['meta']['pretrained_encoder']
    retrain = args['meta']['retrain']
    backbone = args['meta']['backbone']
    model_name = args['meta']['model_name']
    encoder_im, encoder_lid = init_model(device=device, mode=backbone, model_name=model_name)
    # --

    # Optimization
    epochs = int(args['optimization']['epochs'])
    margin = args['optimization']['margin']
    learning_rate = float(args['optimization']['lr'])
    mode = args['optimization']['mode']
    # --
    loss_func = CL(margin=margin, mode=mode)
    optimizer_im, scheduler_im = init_opt(model=encoder_im, args=args['optimization'])
    optimizer_lid, scheduler_lid = init_opt(model=encoder_lid, args=args['optimization'])
    # --

    if pretrained_encoder:
        pretrained_name = args['meta']['pretrained_name']
        starting_epoch = args['meta']['starting_epoch']

        if starting_epoch == 'latest':
            path_encoders = os.path.join(project_root, f'outputs_gpu/{pretrained_name}/models',
                                         f'{pretrained_name}_contrastive-latest.pth.tar')
        else:
            path_encoders = os.path.join(project_root, f'outputs_gpu/{pretrained_name}/models',
                                         f'{pretrained_name}_contrastive-ep{starting_epoch}.pth.tar')

        encoder_im, encoder_lid, opt_im, opt_lid, epoch = load_checkpoint(r_path=path_encoders,
                                                                          encoder_im=encoder_im,
                                                                          encoder_lid=encoder_lid,
                                                                          opt_im=optimizer_im, opt_lid=optimizer_lid)
        for param_group in opt_im.param_groups:
            print(f"Current learning rate of im/lid optimizer: {param_group['lr']}")

    def save_checkpoint(epoch, curr_loss):
        save_dict = {
            'encoder_im': encoder_im.state_dict(),
            'encoder_lid': encoder_lid.state_dict(),
            'optimizer_im': optimizer_im.state_dict(),
            'optimizer_lid': optimizer_lid.state_dict(),
            'epoch': epoch,
            'train_loss': curr_loss,
            'batch_size': batch_size,
            'lr': learning_rate,
            'train_cls': train_classifier
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    if not pretrained_encoder or retrain:
        for epoch in range(epochs):
            training_loss = 0
            validation_loss = 0

            # Training stage: set the model to training mode
            encoder_im.train()
            encoder_lid.train()

            training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')

            for train_iteration, batch in training_loop:
                optimizer_im.zero_grad()
                optimizer_lid.zero_grad()

                left_img_batch = batch['left_img'].to(device)
                depth_batch = batch['depth'].to(device)
                depth_neg = batch['depth_neg'].to(device)

                stacked_depth_batch, img_batch, label_list, stacked_mask = gen_mixed_data(left_img_batch, depth_batch, depth_neg, device, masking)

                # Prediction & Backpropagation
                pred_im = encoder_im.forward(img_batch)
                pred_lid = encoder_lid.forward(stacked_depth_batch)
                torch.cuda.synchronize()

                # For pixel-wise comparison
                N, C, H, W = img_batch.size()

                # Calculating the loss
                loss = loss_func(output_im=pred_im, output_lid=pred_lid, labels=label_list, model_im=encoder_im, H=H, W=W,
                                 vit=vit, mask=stacked_mask)
                loss.backward()
                optimizer_im.step()
                optimizer_lid.step()

                training_loss += loss.item()

                # Update the progress bar.
                training_loop.set_postfix(acc_train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                                          curr_train_loss="{:.8f}".format(loss))

                # Update the tensorboard logger.
                logger.add_scalar(f'siamese_{save_name}/train_loss', loss.item(), epoch * len(train_loader) + train_iteration)

            training_loss /= len(train_loader)
            logger.add_scalar('training_loss_epoch', training_loss, epoch)
            save_checkpoint(epoch, training_loss)

            # Validation stage
            encoder_im.eval()
            encoder_lid.eval()
            val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

            with torch.no_grad():
                for val_iteration, val_batch in val_loop:

                    left_img_batch = val_batch['left_img'].to(device)
                    depth_batch = val_batch['depth'].to(device)
                    depth_neg = val_batch['depth_neg'].to(device)

                    stacked_depth_val, img_batch, label_val, stacked_mask = gen_mixed_data(left_img_batch, depth_batch, depth_neg, device, masking)

                    pred_im = encoder_im.forward(img_batch)
                    pred_lid = encoder_lid.forward(stacked_depth_val)
                    torch.cuda.synchronize()

                    N, C, H, W = img_batch.size()
                    loss_val = loss_func(output_im=pred_im, output_lid=pred_lid, labels=label_val, model_im=encoder_im,
                                         H=H, W=W, vit=vit, mask=stacked_mask)
                    validation_loss += loss_val.item()

                    # Update the progress bar.
                    val_loop.set_postfix(acc_val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)),
                                         curr_val_loss="{:.8f}".format(loss_val))

                    # Update the tensorboard logger.
                    logger.add_scalar(f'siamese_{save_name}/val_loss', loss_val.item(), epoch * len(val_loader) + val_iteration)

            # Epoch-wise calculation
            validation_loss /= len(val_loader)
            logger.add_scalar('validation_loss_epoch', validation_loss, epoch)
            # Step the scheduler after each epoch
            scheduler_im.step()
            scheduler_lid.step()

    if train_classifier:
        # Load pretrained_encoder
        if retrain or not pretrained_encoder:
            print(f"Used latest path")
            trained_enc_im, trained_enc_lid = init_model(device=device,
                                                         mode=backbone,
                                                         model_name=model_name)
            trained_enc_im, trained_enc_lid, opt_im, opt_lid, epoch = load_checkpoint(latest_path,
                                                                                      encoder_im=trained_enc_im,
                                                                                      encoder_lid=trained_enc_lid,
                                                                                      opt_lid=optimizer_lid,
                                                                                      opt_im=optimizer_im)
        else:
            print(f"Used pretrained encoder in classifier:{path_encoders}")
            pretrained_name = args['meta']['pretrained_name']
            starting_epoch = args['meta']['starting_epoch']

            if starting_epoch == 'latest':
                path_encoders = os.path.join(project_root, f'outputs_gpu/{pretrained_name}/models',
                                             f'{pretrained_name}_contrastive-latest.pth.tar')
            else:
                path_encoders = os.path.join(project_root, f'outputs_gpu/{pretrained_name}/models',
                                             f'{pretrained_name}_contrastive-ep{starting_epoch}.pth.tar')

            trained_enc_im, trained_enc_lid, opt_im, opt_lid, epoch = load_checkpoint(r_path=path_encoders,
                                                                              encoder_im=encoder_im,
                                                                              encoder_lid=encoder_lid,
                                                                              opt_im=optimizer_im,
                                                                              opt_lid=optimizer_lid)

        trained_enc_im.to(device).eval()
        trained_enc_lid.to(device).eval()

        model_cls = classifier_head(model_im=trained_enc_im, model_lid=trained_enc_lid, model_name=model_name).to(device)
        # -

        # Optimizer
        learning_rate = float(args['optimization_cls']['lr'])
        epochs = int(args['optimization_cls']['epochs'])
        optimizer, scheduler = init_opt(model=model_cls, args=args['optimization_cls'])
        loss_func = nn.BCELoss()
        # -
        latest_path_cls = os.path.join(project_root, 'outputs/models', f'{save_name}_{tag_cls}-latest.pth.tar')
        save_path_cls = os.path.join(project_root, 'outputs/models', f'{save_name}_{tag_cls}' + '-ep{epoch}.pth.tar')
        logger = tb_logger(root=project_root, args=args['logging_cls'], name=save_name)

        def save_checkpoint_cls(epoch, curr_loss):
            save_dict = {
                'classifier': model_cls.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': curr_loss,
                'batch_size': batch_size,
                'lr': learning_rate,
                'train_cls': train_classifier
            }
            torch.save(save_dict, latest_path_cls)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path_cls.format(epoch=f'{epoch + 1}'))

        for epoch in range(epochs):
            cls_training_loss = 0
            cls_validation_loss = 0

            # Training stage: set the model to training mode
            model_cls.train()

            # Create a progress bar for the training loop.
            training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')

            for train_iteration, batch in training_loop:
                optimizer.zero_grad()  # Reset the gradients

                left_img_batch = batch['left_img'].to(device)
                depth_batch = batch['depth'].to(device)
                depth_neg = batch['depth_neg'].to(device)
                
                stacked_depth_batch, img_batch, label_list, stacked_mask = gen_mixed_data(left_img_batch, depth_batch, depth_neg, device, masking)

                N, C, H, W = img_batch.size()

                pred_cls = model_cls.forward(image=img_batch, lidar=stacked_depth_batch, H=H, W=W)
                pred_cls = pred_cls.squeeze(dim=1)

                loss = loss_func(pred_cls, label_list)
                loss.backward()
                optimizer.step()

                cls_training_loss += loss.item()

                # Update the progress bar.
                training_loop.set_postfix(acc_train_loss="{:.8f}".format(cls_training_loss / (train_iteration + 1)),
                                          curr_train_loss="{:.8f}".format(loss))

                # Update the tensorboard logger.
                logger.add_scalar(f'classifier_{save_name}/train_loss', loss.item(),
                                    epoch * len(train_loader) + train_iteration)

            cls_training_loss /= len(train_loader)
            logger.add_scalar('training_cls_loss_epoch', cls_training_loss, epoch)
            save_checkpoint_cls(epoch, cls_training_loss)

            # Validation stage
            model_cls.eval()
            val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

            with torch.no_grad():
                for val_iteration, val_batch in val_loop:
                    left_img_batch = val_batch['left_img'].to(device)
                    depth_batch = val_batch['depth'].to(device)
                    depth_neg = val_batch['depth_neg'].to(device)
                    
                    stacked_depth_batch, img_batch, label_val, stacked_mask = gen_mixed_data(left_img_batch, depth_batch, depth_neg, device, masking)

                    N, C, H, W = img_batch.size()
                    pred_cls = model_cls.forward(image=img_batch, lidar=stacked_depth_batch, H=H, W=W).squeeze(dim=1)

                    loss_val = loss_func(pred_cls, label_val)
                    cls_validation_loss += loss_val.item()

                    # Update the progress bar.
                    val_loop.set_postfix(val_loss="{:.8f}".format(cls_validation_loss / (val_iteration + 1)),
                                         curr_val_loss="{:.8f}".format(loss_val))

                    # Update the tensorboard logger.
                    logger.add_scalar(f'classifier_{save_name}/val_loss', loss_val.item(), epoch * len(val_loader) + val_iteration)

            scheduler.step()
            cls_validation_loss /= len(val_loader)
            logger.add_scalar('validation_cls_loss_epoch', cls_validation_loss, epoch)
            # torch.cuda.empty_cache()

    else:
        print("Not training the classifier.")

        
