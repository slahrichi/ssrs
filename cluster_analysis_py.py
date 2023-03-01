# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os
from tqdm import tqdm


# %%
def clustering(data_file_list, soft_cluster=False):
    for data_file in data_file_list:
        print('working for feature file ', data_file)
        data = pd.read_feather(data_file)

        feather_array = data[['features_{}'.format(i) for i in range(2048)]].values

        # Start the clustering
        random_state = 42
        cluster_list = [10, 20, 30, 40, 50]#, 100]#, 200]
        cluster_dict = {}
        if soft_cluster:        # If this is a soft clustering, using GMM
            for n_cluster in cluster_list:
                gmm = GaussianMixture(n_components=n_cluster, random_state=random_state).fit(feather_array)
                y_pred = gmm.predict_proba(feather_array)
                np.save(data_file.replace('.feather', '_soft_cluster_{}.npy'.format(n_cluster)), y_pred)
            continue;

        for n_cluster in cluster_list:
            y_pred = KMeans(n_clusters=n_cluster, random_state=random_state).fit_predict(feather_array)
            cluster_dict[n_cluster] = np.copy(y_pred)
            print('clusting for {} finished'.format(n_cluster))

        # Put the clustered index back into the data frame
        for n_cluster in cluster_list:
            data['cluster_index_{}'.format(n_cluster)] = cluster_dict[n_cluster]

        # Save the data frame to a feather format
        data.to_feather(data_file)


def mean_white_balance(img):
    """
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    """
    # 读取图像
    b, g, r = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img

def increase_brightness(img, value=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
    
def visualize_images_batch(img_list ,n_by_n=3, save_name=None, dpi=1500):
    """
    Visualize the images into a big batch
    """
    assert len(img_list) == n_by_n * n_by_n, 'Your number of img provides is {} but the plotting wants to do {} by {}'.format(len(img_list), n_by_n, n_by_n)
    f = plt.figure(figsize=[15,15])
    z = 1
    for i in range(n_by_n):
        for j in range(n_by_n):
            ax = plt.subplot(int('{}{}{}'.format(n_by_n, n_by_n, z)))
            img = mpimg.imread(img_list[z - 1])
            # Version 1 white balance
            # Version 2 white balance
            result = increase_brightness(img)
            result = mean_white_balance(result)
            # image_norm = cv2.normalize(result, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)

            imgplot = plt.imshow(result)
            plt.axis('off')
            z += 1
    plt.tight_layout()
    plt.show()
    if save_name is not None:
        plt.savefig(save_name)
    plt.clf()
    
def visualize(clustered_feature_file, save_dir):
    data = pd.read_feather(clustered_feature_file)
    cluster_list = [10, 20, 30, 40, 50, 100, 200]
    for n_cluster in cluster_list:
        save_folder = os.path.join(save_dir, 'cluster_{}'.format(n_cluster))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for i in tqdm(range(n_cluster)):
            img_list = data.loc[data['cluster_index_{}'.format(n_cluster)] == i][:9]
            img_list = list(img_list['img'].astype('str'))
            if len(img_list) != 9:
                print('there are only {} images in {} cluster of total cluster number {}'.format(len(img_list), i, n_cluster))
                continue;
            visualize_images_batch(img_list, save_name=os.path.join(save_folder, 'cluster_index_{}.png'.format(i)))
            # quit()

if __name__ == '__main__':
    # The clustering piece
    #feature_model_list = ['resnet50_imagenet', 'resnet50_swav_imagenet', 'resnet50_swav_RS']    # Without the CO2 version
    #feature_model_list = ['resnet50_imagenet', 'resnet50_swav_imagenet', 'resnet50_CO2', 'resnet50_swav_RS']
    feature_model_list = ['resnet50_CO2']
    for feature_model in feature_model_list:
        data_file_list = [
        # '/data/shared/satellite_derived_feature/MA/MA_features_{}.feather'.format(feature_model),
        #                 '/data/shared/satellite_derived_feature/MI/MI_features_{}.feather'.format(feature_model)]
                         #'/data/shared/satellite_derived_feature/Mexico/Mexico_features_{}.feather'.format(feature_model),]
                         '/data/shared/satellite_derived_feature/Indonesia/Indonesia_features_{}.feather'.format(feature_model),]

        #data_file_list = ['/data/shared/satellite_derived_feature/MA/MA_features_{}.feather'.format(feature_model)]
        clustering(data_file_list, soft_cluster=True)
        clustering(data_file_list, soft_cluster=False)

    #########################################################
    # The visualizaation piece
    # feature_file = '/data/shared/satellite_derived_feature/MI/MI_features_resnet50_imagenet.feather'
    # save_dir = '/data/shared/cluster_analysis/MI'

    # feature_file = '/data/shared/satellite_derived_feature/MA/MA_features_resnet50_imagenet.feather'
    # save_dir = '/data/shared/cluster_analysis/MA'

    # feature_file = '/data/shared/satellite_derived_feature/Mexico/Mexico_features_resnet50_imagenet.feather'
    # save_dir = '/data/shared/cluster_analysis/Mexico'

    # visualize(feature_file,  save_dir)
