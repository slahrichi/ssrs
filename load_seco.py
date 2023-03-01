import torch
from models.base import resnet
from torchvision.models import resnet50 as r50
from collections import OrderedDict
import re

model_path = "/home/sl636/seasonal-contrast/checkpoints/seco_resnet50_1m.ckpt"
checkpoint = torch.load(model_path)
checkpoint_dict =  checkpoint["state_dict"]
#print(checkpoint_dict.keys())

# for k in list(checkpoint_dict.keys()):
#     if k.startswith("encoder_q") :
#         checkpoint_dict["layer"+k[len("encoder_q."):]] = checkpoint_dict[k]
#     del checkpoint_dict[k]

# print(checkpoint_dict)
# model = resnet.resnet50(inter_features=True)
# model.load_state_dict(checkpoint_dict, strict=True)

new_state_dict = OrderedDict()

for key in list(checkpoint_dict.keys()):
    if key.startswith("encoder_q"):
        new_key = re.sub("encoder_q.","",key)
        new_key = re.sub("^0(?=\.weight)","conv1",new_key)
        new_key = re.sub("^1(?=\.[a-z_]+)","bn1",new_key)
        new_key  = re.sub("^4(?=\.\d\.(conv|bn|downsample))","layer1",new_key )
        new_key  = re.sub("^5(?=\.\d\.(conv|bn|downsample))","layer2",new_key )
        new_key  = re.sub("^6(?=\.\d\.(conv|bn|downsample))","layer3",new_key )
        new_key  = re.sub("^7(?=\.\d\.(conv|bn|downsample))","layer4",new_key )
        new_state_dict[new_key] = checkpoint_dict[key] 

model = r50(pretrained=True)
model.load_state_dict(new_state_dict)
torch.save(model.state_dict(), "seco_resnet50_2.pt")