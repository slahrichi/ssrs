import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_folder = "/home/sl636/seasonal-contrast/2750/"

# Get a list of all the subfolders (categories) in the data folder
subfolders = os.listdir(data_folder)

# Create an empty list to store the image filepaths and their corresponding labels
image_filepaths = []
labels = []

# Loop over each subfolder (category) and get the filepaths of all images in that folder
for i, subfolder in tqdm(enumerate(subfolders)):
    subfolder_path = os.path.join(data_folder, subfolder)
    image_names = os.listdir(subfolder_path)
    image_paths = [os.path.join(subfolder_path, name) for name in image_names]
    
    # Add the image filepaths and their corresponding labels to the lists
    image_filepaths.extend(image_paths)
    labels.extend([i]*len(image_paths))

# Convert the labels to numpy array
labels = np.array(labels)

# Perform a stratified train-test split
train_filepaths, test_filepaths, train_labels, test_labels = train_test_split(image_filepaths, labels, 
                                                                              test_size=0.2, random_state=42,
                                                                              stratify=labels)

with open("/home/sl636/seasonal-contrast/datasets/eurosat/train.txt", "w") as file:
    for path in train_filepaths:
        file.write(path.split("/")[-1]+"\n")

    
with open("/home/sl636/seasonal-contrast/datasets/eurosat/val.txt", "w") as file:
    for path in test_filepaths:
        file.writelines(path.split("/")[-1]+"\n")