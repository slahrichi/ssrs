import logging

import torch
from torchvision.models import resnet50 as r50
from copy import deepcopy

import re
from collections import OrderedDict
from .base import resnet


def load(encoder_name):
    if encoder_name == "swav":
        print("Loading swav-imagenet pretrained weights.")
        return _load_swav()
    elif encoder_name == "none":
        print("Loading encoder with no pretrained weights.")
        return _load_base()
    elif encoder_name == "imagenet":
        print("Loading supervised ResNet model.")
        return _load_imagenet()
    elif encoder_name == "swav-climate+":
        print("Loading swav-climate+ pretrained weights")
        return _load_swav_pretrained("/home/sl636/swav/experiments/indep/swav/climate+/swav-climate+.pt")
    elif encoder_name == "seco":
        return _load_seco("/home/sl636/seasonal-contrast/checkpoints/seco_resnet50_1m.ckpt")
    elif encoder_name.startswith("swav_climate+_ep"):
        n = encoder_name[len("swav_climate+_ep"):]
        if n in ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "99"]:
            print(f"Loading swav-climate+_ep{n} pretrained weights")
            return _load_swav_pretrained(f"/home/sl636/ssrs/models/swav/ckp-{n}.pt")
        else:
            print("Couldn't find encoder: ", encoder_name)
            return
    elif encoder_name.startswith("swav-climate+only-ep"):
        n = encoder_name[len("swav-climate+only-ep"):]
        if n in ["0","25", "50", "75", "100", "125", "150", "175", "200"]:
            print(f"Loading swav-climate+only-{n} pretrained weights")
            return _load_swav_pretrained(f"/home/sl636/ssrs/models/swav/climate+only/models/ckp-{n}.pt")
        else:
            print("Couldn't find encoder: ", encoder_name)
            return
    else:
        logging.error(f"Encoder {encoder_name} not implemented.")
        raise NotImplementedError


def _load_seco(model_path):
    checkpoint = torch.load(model_path)
    checkpoint_dict =  checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for key in list(checkpoint_dict.keys()):
        if key.startswith("encoder_q"):
            new_key = re.sub("encoder_q.","",key)
            new_key = re.sub("^0(?=\.weight)","conv1",new_key)
            new_key = re.sub("^1(?=\.[a-z_]*)","bn1",new_key)
            new_key  = re.sub("^4(?=\.\d\.(conv|bn|downsample))","layer1",new_key )
            new_key  = re.sub("^5(?=\.\d\.(conv|bn|downsample))","layer2",new_key )
            new_key  = re.sub("^6(?=\.\d\.(conv|bn|downsample))","layer3",new_key )
            new_key  = re.sub("^7(?=\.\d\.(conv|bn|downsample))","layer4",new_key )
            new_state_dict[new_key] = checkpoint_dict[key] 

    model = resnet.resnet50(inter_features=True)
    model.load_state_dict(new_state_dict)
    
    return model

def _load_swav_pretrained(model_path):
    """
    This function loads the swav encoder pretrained on the solar training
    dataset.
    """
    state_dict = torch.load(model_path)
    base_model = resnet.resnet50(inter_features=True)
    base_model.load_state_dict(state_dict)

    return base_model


def _load_imagenet():
    """
    This model loads the weights from the SwAV model and places them
    onto this version of the ResNet model which allows the layers
    to be passed forward 

    """
    model = r50(pretrained=True)
    return _append_state_dict_to_resnet(model.state_dict())


def _load_base():
    # This only loads the base encoder model
    # with no pretrained weights
    base_model = resnet.resnet50(inter_features=True)
    return base_model


def _load_swav():
    """
    This model loads the weights from the SwAV model and places them
    onto this version of the ResNet model which allows the layers
    to be passed forward
    """
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    return _append_state_dict_to_resnet(model.state_dict())


# def _load_swav_b2():
#     """
#     This model loads the weights from the SwAV model that's trained
#     on the target data using its mean and standard deviation for the 
#     normalization scheme and places them onto this version of the 
#     ResNet model which allows the layers to be passed forward 

#     """
#     model = torch.load("./swav-models/swav-b1.pt")
#     return _append_state_dict_to_resnet_2(model)


# def _append_state_dict_to_resnet_2(state_dict):
#     # Instantiate the version of ResNet that we want
#     # and load the weights on top of this model.
#     # For semantic segmentation, we need inter_features
#     # to be true.
#     #
#     # As we add more functionality, this piece of code
#     # will need to change.
#     base_model = resnet.resnet50(inter_features=True)
#     base_model.load_state_dict(state_dict)
#     return base_model


def _append_state_dict_to_resnet(state_dict):

    # Remove keys that we don't need
    state_dict.pop("fc.bias")
    state_dict.pop("fc.weight")

    # Instantiate the version of ResNet that we want
    # and load the weights on top of this model.
    # For semantic segmentation, we need inter_features
    # to be true.
    #
    # As we add more functionality, this piece of code
    # will need to change.
    base_model = resnet.resnet50(inter_features=True)
    base_model.load_state_dict(state_dict)
    return base_model
