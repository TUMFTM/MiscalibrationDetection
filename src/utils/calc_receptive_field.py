import torch
import torch.nn as nn
import torchvision


class PixelwiseFeatureMaps:
    def __init__(self, model, embeddings_value, input_image_size):
        self.model = model
        self.embeddings_value = embeddings_value.cuda() if torch.cuda.is_available() else embeddings_value
        self.input_image_size = input_image_size  # (H, W) of the input image

    def _extract_params(self):
        def extract_from_layer(layer):
            params = []
            if isinstance(layer, nn.Conv2d):
                params.append({
                    'layer_type': 'conv',
                    'kernel_size': layer.kernel_size if isinstance(layer.kernel_size, tuple) else (
                    layer.kernel_size, layer.kernel_size),
                    'stride': layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride),
                    'padding': layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                })
            elif isinstance(layer, nn.MaxPool2d):
                params.append({
                    'layer_type': 'pool',
                    'kernel_size': layer.kernel_size if isinstance(layer.kernel_size, tuple) else (
                    layer.kernel_size, layer.kernel_size),
                    'stride': layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride),
                    'padding': layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                })
            elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU, nn.Identity)):
                # Skip layers that don't affect the receptive field directly
                pass
            elif isinstance(layer, nn.Sequential) or isinstance(layer, torchvision.models.resnet.BasicBlock):
                for name, sub_layer in layer.named_children():
                    params.extend(extract_from_layer(sub_layer))
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
            return params

        layers = list(self.model.children())
        params = []
        for layer in layers:
            params.extend(extract_from_layer(layer))

        return params

    def calculate_receptive_fields(self):
        layer_params = self._extract_params()
        # print(f"layerparams:{layer_params}")
        H, W = self.input_image_size

        # Initialize receptive field parameters
        rf_size = torch.ones((H, W), device=self.embeddings_value.device)
        stride = torch.ones((H, W), device=self.embeddings_value.device)
        start_x = torch.arange(H, device=self.embeddings_value.device).unsqueeze(1).repeat(1, W).float()
        start_y = torch.arange(W, device=self.embeddings_value.device).unsqueeze(0).repeat(H, 1).float()

        for param in layer_params:
            if param['layer_type'] == 'conv' or param['layer_type'] == 'pool':
                k, s, p = param['kernel_size'][0], param['stride'][0], param['padding'][0]
                rf_size += (k - 1) * stride
                stride *= s
                start_x = start_x * s - p
                start_y = start_y * s - p

        end_x = start_x + rf_size
        end_y = start_y + rf_size

        return start_x.long(), end_x.long(), start_y.long(), end_y.long()

    def assign_embedding_value(self):
        N, C, H, W = self.embeddings_value.shape
        input_H, input_W = self.input_image_size
        result = torch.zeros((N, C, input_H, input_W), device=self.embeddings_value.device)
        count = torch.zeros((N, 1, input_H, input_W), device=self.embeddings_value.device)

        start_x, end_x, start_y, end_y = self.calculate_receptive_fields()

        for i in range(H):
            for j in range(W):
                sx, ex = max(0, start_x[i, j]), min(input_H, end_x[i, j])
                sy, ey = max(0, start_y[i, j]), min(input_W, end_y[i, j])

                result[:, :, sx:ex, sy:ey] += self.embeddings_value[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                count[:, :, sx:ex, sy:ey] += 1

        # Avoid division by zero
        count = count + (count == 0).float()
        result /= count

        return result

    def assign_embedding_layerwise(self):
        layer_params = self._extract_params()
        H, W = self.input_image_size

        # Initialize receptive field parameters
        rf_size = torch.ones((H, W), device=self.embeddings_value.device)
        stride = torch.ones((H, W), device=self.embeddings_value.device)
        start_x = torch.arange(H, device=self.embeddings_value.device).unsqueeze(1).repeat(1, W).float()
        start_y = torch.arange(W, device=self.embeddings_value.device).unsqueeze(0).repeat(H, 1).float()

        N, C, H_emb, W_emb = self.embeddings_value.shape
        result = torch.zeros((N, C, H, W), device=self.embeddings_value.device)
        count = torch.zeros((N, 1, H, W), device=self.embeddings_value.device)

        for param in layer_params:
            if param['layer_type'] == 'conv' or param['layer_type'] == 'pool':
                k, s, p = param['kernel_size'][0], param['stride'][0], param['padding'][0]
                
                rf_size += (k - 1) * stride
                stride *= s
                start_x = start_x * s - p
                start_y = start_y * s - p
                
                end_x = start_x + rf_size
                end_y = start_y + rf_size

                for i in range(H_emb):
                    for j in range(W_emb):
                        sx, ex = max(0, start_x[i, j]), min(H, end_x[i, j])
                        sy, ey = max(0, start_y[i, j]), min(W, end_y[i, j])

                        result[:, :, sx:ex, sy:ey] += self.embeddings_value[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                        count[:, :, sx:ex, sy:ey] += 1

        # Avoid division by zero
        count = count + (count == 0).float()
        result /= count

        return result