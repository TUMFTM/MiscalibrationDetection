import os.path
import torch
from src.datasets.dataloader.dataset_2D import create_dataloaders
import multiprocessing
from cka_modified import CKA
# from torch_cka import CKA


def compute_and_save_cka_heatmap(model1, model2, dataloader1, dataloader2, tag_1, tag_2, title,
                                 save_path='cka_heatmap.png', show_plot=False):
    # Function to extract layer names from the model
    def get_layer_names(model):
        layer_names = []
        for name, module in model.named_modules():
            layer_names.append(name)
        return layer_names

    # Set the model to evaluation mode
    model1.eval()
    model2.eval()

    # Automatically get the layer names for both models
    layer_names_1 = get_layer_names(model1)
    layer_names_2 = get_layer_names(model2)
    if layer_names_1 == layer_names_2:
        print("The layer names are identical.")
    else:
        print("The layer names differ.")

    # Initialize CKA with unique model names

    # cka = CKA(
    #     model1=model1,
    #     model2=model2,
    #     model1_name=tag_1,  # Provide unique names
    #     model2_name=tag_2,
    #     model1_layers=layer_names_1,
    #     model2_layers=layer_names_2,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    cka = CKA(
        model1=model1,
        model2=model2,
        model1_name=tag_1,  # Provide unique names
        model2_name=tag_2,
        model1_layers=layer_names_1,
        model2_layers=layer_names_2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Perform CKA comparison on the dataloader
    cka.compare(dataloader1=dataloader1, dataloader2=dataloader2)

    cka.plot_results(save_path=save_path, show_plot=show_plot, title=title)

    print(f"CKA heatmap saved to: {save_path}")

def cka_analysis(data_root, output_dir, model_1, model_2, tag_1, tag_2, title, perturbation_eval, loader, augmentation, show_plot=False, crossmodel=False):

    batch_size = 64
    num_cores = min(multiprocessing.cpu_count(), 64)
    dataloader_im, dataloader_lid, dataloader_neg = create_dataloaders(root=data_root, perturb_filenames=perturbation_eval, mode='test',
                                                                       batch_size=batch_size, num_cores=num_cores, loader=loader, augmentation=augmentation)

    if 'Image' in tag_1:
        dataloader_1 = dataloader_im
        save_1 = 'im'
    elif 'Calibrated' in tag_1:
        dataloader_1 = dataloader_lid
        save_1 = 'lid'
    elif 'Miscalibrated' in tag_1:
        dataloader_1 = dataloader_neg
        save_1 = 'neg'
    else:
        assert False, f"Error: tag_1: {tag_1} must contain 'Image' or 'LiDAR'"

    if 'Image' in tag_2:
        dataloader_2 = dataloader_im
        save_2 = 'im'
    elif 'Calibrated' in tag_2:
        dataloader_2 = dataloader_lid
        save_2 = 'lid'
    elif 'Miscalibrated' in tag_2:
        dataloader_2 = dataloader_neg
        save_2 = 'neg'
    else:
        assert False, f"Error: tag_2: {tag_2} must contain 'Image' or 'Calibrated' or 'Miscalibrated'"

    # Check for NaNs in layer outputs
    def check_for_nan(module, input, output):
        if torch.isnan(output).any():
            print(f"NaN detected in module: {module}")

    # Register hooks to check for NaNs
    for name, module in model_1.named_modules():
        module.register_forward_hook(check_for_nan)
    for name, module in model_2.named_modules():
        module.register_forward_hook(check_for_nan)

    save_path = os.path.join(output_dir, f'cka_analysis_{save_1}_{save_2}.png')

    if crossmodel:
        dataloader_2 = None

    # Call the function to compute and save the CKA heatmap between two models
    compute_and_save_cka_heatmap(model_1, model_2, dataloader_1, dataloader_2, tag_1, tag_2, title,
                                 save_path=save_path, show_plot=show_plot)

