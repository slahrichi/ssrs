import numpy as np
import os
from PIL import Image
import time
from tqdm import tqdm

def calculate_mean_std(img_path):
    parts = os.walk(img_path)
    images = []
    for root, dirs, files in parts:
        for name in tqdm(files):
            images.append(os.path.join(root,name))
    mean = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])
    std_temp = np.array([0., 0., 0.])
    start_time = time.time()
    for image in tqdm(images):
        np_img = np.array(Image.open(image))
        np_img = np_img / 255.
        for i in range(3):
            mean[i] += np.mean(np_img[:,:,i])
    mean = mean / len(images)
    print(f"Mean is {mean}")

    for image in tqdm(images):
        np_img = np.array(Image.open(image))
        np_img = np_img / 255.
        for i in range(3):
            std_temp[i] += ((np_img[:,:,i] - mean[i])**2).sum() / (np_img.shape[0]*np_img.shape[1]) 
    std = np.sqrt(std_temp/len(images)) 
    print(f"Std is {std}")
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time} s to calculate the mean & the std of {len(images)}")
    with open("geonet_norm.txt", "w+") as output:
        output.write(f"Mean is {mean}" + "\n" + f"Std is {std}")
    return mean, std

img_path = "/shared/data/cleaned_images10M"

calculate_mean_std(img_path)