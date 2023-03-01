import os
from PIL import Image
from torch.utils.data import Dataset

class EurosatDataset(Dataset):
    def __init__(self, image_path, path, size, split, transform=None, augmentations=None):
        self.image_path = image_path
        self.path = path
        self.size = size
        self.split = split
        self.transform = transform
        self.aug = augmentations


        with open(os.path.join(self.path, f'{size}_{split}.txt')) as f:
            filenames = f.read().splitlines()
        
        self.classes = sorted([d.name for d in os.listdir(self.image_path)])
        self.class_to_idx = {cls_name:i for i, cls_name in enumerate(self.classes)}

        self.images = []
        for fn in filenames:
            cls_name = fn.split("_")[0]
            self.images.append(os.path.join(os.path.join(self.image_path, cls_name), fn))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Load the image
        img_name = self.images[idx]
        img = Image.open(img_name)
        target = self.class_to_idx[img_name.split("_")[0]]

        # We still need to transform to a tensor and normalize.
        if self.transform:
            img = self.transform(img)

        return img, target
