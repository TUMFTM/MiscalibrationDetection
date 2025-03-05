import torch
import torch.nn as nn


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # Freeze

def get_last_layer_channels(model):
    # Get the last block in the model
    last_block = list(model.blocks.children())[-1]

    # Get the last BasicBlock in the last block
    last_basic_block = list(last_block.children())[-1]

    # The last BasicBlock should have a conv2 layer, which is the last convolutional layer
    last_conv_layer = last_basic_block.conv2

    # Return the number of output channels of the last convolutional layer
    return last_conv_layer.out_channels

class classifier_head(nn.Module):
    def __init__(self, model_im, model_lid, model_name=None):
        super().__init__()
        self.model_im = model_im
        self.model_lid = model_lid
        input_channel = get_last_layer_channels(self.model_im) + get_last_layer_channels(self.model_lid)
        first_channel = input_channel*2

        if model_name == 'resnet18_small':
            self.classifier_layers = nn.Sequential(
                nn.Conv2d(input_channel, first_channel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(first_channel),
                nn.ReLU(),
                nn.Conv2d(first_channel, 1024, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), # output: (N, 1024, 1, 1)
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
                )
        elif model_name == 'resnet18_all':
            self.classifier_layers = nn.Sequential(
                # nn.AdaptiveAvgPool2d((1, 1)),
                # nn.Flatten(),
                # nn.Linear(input_channel, 256),
                # nn.BatchNorm1d(256),
                # nn.ReLU(),
                # nn.Linear(256, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                # nn.Linear(128, 1),
                # nn.Sigmoid()
                nn.Conv2d(input_channel, first_channel, kernel_size=3, stride=1, padding=1),  # output: (N,
                nn.BatchNorm2d(first_channel),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(first_channel, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
                )
        elif model_name == 'resnet18_instance_small':
            self.classifier_layers = nn.Sequential(
                nn.Linear(2000, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
                )
        else:
            raise ValueError("Error: starting_epoch couldn't be found")
    def load_classifier_layers(self, state_dict):
        classifier_state_dict = {k.replace('classifier_layers.', ''): v for k, v in state_dict.items() if
                                 k.startswith('classifier_layers.')}
        self.classifier_layers.load_state_dict(classifier_state_dict)

    def forward(self, image, lidar, H, W):
        set_parameter_requires_grad(self.model_im, feature_extracting=True)
        set_parameter_requires_grad(self.model_lid, feature_extracting=True)
        
        self.model_lid.eval()
        self.model_im.eval()
        with torch.no_grad():
            image = self.model_im(image)
            lidar = self.model_lid(lidar)

        z = torch.cat((image, lidar), dim=1)
        z = self.classifier_layers(z)
        return z