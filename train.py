from random import sample
from turtle import Turtle
from typing import Any
from unicodedata import name
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np 
import torchvision
from torchvision import transforms
from torch.utils import data
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import sys
import time
import torch.optim as optim 
from make_graph import *
from tqdm import tqdm
from argparse import ArgumentParser
from efficientnet_pytorch import EfficientNet

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-n', '--no', type=int, default=0, help='research number')
    parser.add_argument('-c', '--out_c', type=int, default=0, help='output number of cluster')
    parser.add_argument('-d', '--disc', type=str, default='D', help='name of drive where data exists')
    return parser.parse_args

def main():
    args = parse_args()

    device = 'cuda'
    input_size = 1000
    batch_size = 8
    num_epochs = 50
    early_point = 10
    LR = 0.001
    lr_step_epoch = 5

    label_alg = 'IIC'
    no = args.no 
    num_class = args.out_c
    disk = args.disc

    label_dir = disc + ''
    print('data_path :', label_alg)

    save_dir = disc + ''

    model_name = ''

    if 'efficient_net' in model_name:
        print(model_name)
        model = EfficientNet.from_pretrained(model_name)
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_class)
    elif model_name == 'resnet34':
        print(model_name)
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(512, num_class),
        )
    elif model_name == 'resnet50':
        print(model_name)
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(2048, num_class),
            nn.Softmax(dim=1),
        )
    else:
        print('model error')
        sys.exit()

    class ImageTransform_train():
        def __init__(self):
            input_size = input_size

            self.data_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Grayscale(num_output_channels=3)
            ])

        def __call__(self, img):
            return self.data_transform(img)
        
    class ImageTransform_val():
        def __init__(self):
            input_size = input_size

            self.data_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize((input_size, input_size)),
                transforms.Grayscale(num_output_channels=3)
            ])
        
        def __call__(self, img):
            return self.data_transform(img)
        
    class My_Dataset(data.Dataset):
        def __init__(self, images, labels, mode):
            super().__init__()

            self.images_path = images
            self.labels = labels
            self.labels = torch.from_numpy(self.labels.astype(np.int64)).clone()

            self.train_transform = ImageTransform_train()
            self.vel_transform = ImageTransform_val()
            self.mode = mode

        def __len__(self):
            return len(self.images_path)
        
        def __getitem__(self, index):

            x_image_path = self.images_path[index]
            x_image =Image.open(x_image_path)

            if self.mode == 'train':
                x_image = self.train_transform(x_image)
            elif self.mode == 'valid':
                x_image = self.vel_transform(x_image)
            else:
                print('dataset mode is not defined')
                sys.exit()

            x_label = self.labels[index]

            x_image = torch.tensor(x_image, dtype=torch.float32)
            x_image = x_image / 255.0
            sample = {
                'image' : x_image,
                'label' : x_label
            }
            return sample
        
    all_images = []
    all_labels = []

    for id in range(num_class):
        glo_dataset = glob(os.path.join(label_dir, str(id), '*jpg'))
        glo_label = np.full(len(glo_dataset), id, dtype='int64')
        all_images = np.concatenate([all_images, glo_dataset], axis=0)
        all_labels = np.concatenate([all_labels, glo_label], axis=0)

    print('all_data_images = ', all_images.shape)
    print('all_label_images = ', all_labels.shape)

    print('-'*50)

    del glo_dataset, glo_label

    X_train, X_test, Y_train, Y_test = train_test_split(all_images, all_labels,
                                                        test_size=0.2,
                                                        random_state=2023,
                                                        stratify=all_labels)
    
    print('X_train images = ', X_train.shape)
    print('Y_train labels = ', Y_train.shape)
    print('X_test images = ', X_test.shape)
    print('Y_test labels = ', Y_test.shape)

    del all_images, all_labels

    train_dataset = My_Dataset(images=X_train, labels=Y_train, mode='train')
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = My_Dataset(images=X_test, labels=Y_test, mode='valid')
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )

    model.to(device)

    all_train_loss = []
    all_train_acc = []
    all_test_loss = []
    all_test_acc = []
    eval_best_loss = 1000
    early_count = 0

    for epoch in range(num_epochs):
        start = time.time()
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        sum_correct = 0
        sum_loss = 0
        sum_total = 0

        model.train()
        count = 0

        for samples in tqdm(train_dataloader):
            #input
            images = samples['image']
            labels = samples['label']

            images = images.to(device)
            labels = labels.to(device)

            #output
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            #loss and acc
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_correct += (predicted == labels).sum().item()
            sum_total += labels.size(0)

            loss.backward()
            optimizer.step()

        print('train_loss = ', sum_loss*batch_size/sum_total)
        print('train_acc = ', sum_correct/sum_total)

        all_train_loss.append(sum_loss*batch_size/sum_total)
        all_train_acc.append(sum_correct/sum_total)

        sum_correct = 0
        sum_loss = 0
        sum_total = 0

        model.eval()

        with torch.no_grad():
            count_t = 0

            for samples in test_dataloader:
                #input
                images = samples['image']
                labels = samples['label']

                images = images.to(device)
                labels = labels.to(device)

                #output
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                #loss and acc
                sum_loss += loss.item()
                _, predicted = outputs.max(1)
                sum_correct += (predicted == labels).sum().item()
                sum_total += labels.size(0)
            
            print('eval_loss = ', sum_loss+batch_size/sum_total)
            print('eval_acc = ', sum_correct/sum_total)

            if (sum_loss*batch_size/sum_total) < eval_best_loss:
                eval_best_loss = (sum_loss*batch_size/sum_total)

                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), save_dir + '/' + model_name + '_{}.pth'.format(no))

                early_count = 0
                print('save now!')

            else:
                early_count += 1
                if early_count % lr_step_epoch == 0:
                    scheduler.step()
                    print('lr update')

            if early_count == early_point:
                print('early stop')
                break

            all_test_loss.append(sum_loss*batch_size/sum_total)
            all_test_acc.append(sum_correct/sum_total)

        elapsed_time = time.time() - start

        print('elapsed_time : {0}'.format(elapsed_time) + '[sec]')
        print('-'*50)

    print('all_train_acc = ', all_train_acc)
    print('all_eval_acc = ', all_test_acc)

    make_2_target_graph(train=all_train_loss, eval=all_test_loss, name='loss',
                        model_name=model_name, out_dir=save_dir)
    
    make_2_target_graph(train=all_train_acc, eval=all_test_acc, name='acc',
                        model_name=model_name, out_dir=save_dir)
    
    all_train_loss = pd.DataFrame(all_train_loss)
    all_train_acc = pd.DataFrame(all_train_acc)
    all_test_loss = pd.DataFrame(all_test_loss)
    all_test_acc = pd.DataFrame(all_test_acc)

    df_csv = pd.concat([all_train_loss, all_train_acc,
                        all_test_loss, all_test_acc], axis=1)
    
    df_csv.colums = ['train_loss', 'train_acc', 'test_loss', 'test_acc']

    print(df_csv.head())

    df_csv.to_csv(save_dir+'/'+model_name+'.csv')

if __name__ == '__main__':
    main()