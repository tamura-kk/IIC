import torch
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import random

def make_filepath_list(file_path):
    """
    学習データ、検証データそれぞれのファイルへのパスを格納したリストを返す
    
    Returns
    -------
    train_file_list: list
        学習データファイルへのパスを格納したリスト
    valid_file_list: list
        検証データファイルへのパスを格納したリスト
    """
    train_file_list = []
    valid_file_list = []
    classes = []

    for top_dir in os.listdir(file_path):
        classes += top_dir
        file_dir = os.path.join(file_path, top_dir)
        file_list = os.listdir(file_dir)
        random.shuffle(file_list)

        # 各データごとに8割を学習データ、2割を検証データとする
        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        train_file_list += [os.path.join(file_path, top_dir, file).replace('\\', '/') for file in file_list[:num_split]]
        valid_file_list += [os.path.join(file_path, top_dir, file).replace('\\', '/') for file in file_list[num_split:]]
    
    return train_file_list, valid_file_list, classes

#---------------データの前処理---------------#
# 画像データへのファイルパスを格納したリストを取得する
file_path = 'C:/Users/nougata-share-pc/Desktop/Tamura/IIC/cluster_data'
train_file_list, valid_file_list, classes = make_filepath_list(file_path)

print('学習データ数 : ', len(train_file_list))
print('検証データ数 : ', len(valid_file_list))
print(int(len(classes)))