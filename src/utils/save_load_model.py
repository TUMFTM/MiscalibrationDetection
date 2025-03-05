import os
import pickle
import torch

from src.models.mm_siamese import resnet18_2B_lid
from src.models.mm_siamese import resnet18_2B_im
from src.models.classifier_head import classifier_head


def load_model_lidar(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = resnet18_2B_lid()
    model.load_state_dict(model_dict["state_dict"])
    print("model is loaded")
    return model


def load_model_cls(model_path, model_im, model_lid, pixel_wise):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]

    model = classifier_head(model_im=model_im, model_lid=model_lid, model_name=pixel_wise)
    model.load_classifier_layers(model_dict["state_dict"])

    print("model is loaded")
    return model


def load_model_img(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = resnet18_2B_im()
    model.load_state_dict(model_dict["state_dict"])
    print("model is loaded")
    return model


def save_model(model, file_name, directory="outputs/models"):

    model = model.cpu()
    model_dict = {"cifar_classification_ptl": {"state_dict": model.state_dict()}}
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("dir created")
    pickle.dump(model_dict, open(os.path.join(directory, file_name), 'wb', 4))
    print("Model's parameters saved")