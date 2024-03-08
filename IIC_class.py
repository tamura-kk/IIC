import random
import numpy as np
import torch
from argparse import ArgumentParser
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os
import shutil
from glob import glob
from torchsummary import summary
import torch.optim as optim
from tqdm import tqdm
import argparse
from config import get_config

from II2.II2_loss import *
from II2.II2_transform import *

import matplotlib.pyplot as plt

from models.ModelDefine import NET
import pandas as pd
import multiprocessing
import warnings
warnings.simplefilter('ignore')

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# def get_args():
#     parser = ArgumentParser(
#         description='This is sample argparse script')
#     parser.add_argument('-n', '--no', default=0,
#                         type=int, help='This is name')
#     parser.add_argument('-c', '--out_c', default=0,
#                         type=int, help='This is name')
#     return parser.parse_args()

def parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def resize_image(config):
    if config.DATA.COLOR == 'color':
        transform = transforms.Compose([
            transforms.Resize(config.DATA.IMAGE_SIZE),
            transforms.PILToTensor(),
        ])
    
    elif config.DATA.COLOR == 'gray':
        transform = transforms.Compose([
            transforms.Resize(config.DATA.IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ])

    elif config.DATA.COLOR == 'colornorm':
        transform = transforms.Compose([
            transforms.Resize(config.DATA.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.728, 0.514, 0.708],
                                 std=[0.380, 0.446, 0.471])
        ])

    return transform

class My_Dataset(Dataset):
    def __init__(self,color, train_data, transforms=None):
        self.path = train_data
        self.transforms = transforms
        self.color = color

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        image_path = self.path[index]
        image = Image.open(image_path)

        if self.transforms:
            image = self.transforms(image)

        image = torch.tensor(image, dtype=torch.float32)

        g_image, _, __ = random_affine(image.to('cuda'))
        g_image = g_image.to('cpu')

        del _, __

        if self.color != 'colornorm':
            image = image / 255.0
            g_image = g_image / 255.0

        return {
            'images' : image,
            'g_images' : g_image
        }
        
def make_graph(xlabel, ylabel, List, List2, path, number):

        fig = plt.figure()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()

        idx = List.index(ylabel)
        List2 = np.array(List2)

        plt.plot(List2[:, idx], label=ylabel)
        plt.legend()

        fig.savefig(path + str(number) + '_' + ylabel + '.png')

def main(config):

    start = time.time()

    print('use number of gpus = ', torch.cuda.device_count())
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    print('device_name ----> ', device)

    no = config.MODEL.NUMBER
    out_c = config.DATA.OUTPUT_CHANNEL
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE

    print('experiment no. ---->', no)
    print('data_type ---->', data_type)
    
    base_path = os.path.join('result',  str(data_type) ,str(model_name), str(no), 'train', 'class_model')
    
    os.makedirs(base_path, exist_ok=True)
    
    net_path = base_path

    shutil.copyfile('config.py', os.path.join(base_path,'config_{}.py'.format(no)))

    f = open(os.path.join(base_path,'conditions.txt'), 'w')

    datalist = ['Experiment No. : {}\n'.format(no),
                'Epochs : {}\n'.format(config.TRAIN.EPOCHS),
                'Batch size : {}\n'.format(config.DATA.BATCH_SIZE),
                'Learning rate : {}\n'.format(config.TRAIN.LR),
                'Image size : {}\n'.format(config.DATA.IMAGE_SIZE),
                'Class : {}\n'.format(out_c),
                'Model name : {}\n'.format(config.MODEL.TYPE),
                'lr_step : {}\n'.format(config.TRAIN.LR_STEP),
                'color : {}\n'.format(config.DATA.COLOR),
                'data_path : {}\n'.format(config.DATA.TRAIN_DATA_PATH),
                'data_type :  {}\n'.format(config.DATA.TYPE),
                '-'*50, '\n' ]
    
    f.writelines(datalist)
    f.close()

    train_data = glob(config.DATA.TRAIN_DATA_PATH)
    print('number of images : ', len(train_data))

    origin_dataset = My_Dataset(
       config.DATA.COLOR, train_data, transforms=resize_image(config)
    )
    
    origin_loader = DataLoader(
        origin_dataset, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
        shuffle=True, worker_init_fn=seed_worker
    )

    print(model_name)

    net = NET(config)
    net = net.to(device)

    if config.MODEL.CHECKPOINT == 'True':
        checkpoint = torch.load(os.path.join(net_path,'checkpoint_best.bin'))
        net.load_state_dict(checkpoint['model'])
        random.setstate(checkpoint['random'])
        np.random.set_state(checkpoint['np_random'])
        torch.set_rng_state(checkpoint['torch'])
        torch.random.set_rng_state(checkpoint['torch_random'])
        torch.cuda.set_rng_state(checkpoint['cuda_random'])


    summary(net, (3, config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE))

    optimizer = optim.Adam(net.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    net.train()

    train_loss_value = []

    print('-'*15)
    print('-------start!!-------')
    print('-'*15)

    early_count = 0
    min_avg_loss = float('inf')

    for epoch in range(config.TRAIN.EPOCHS):
        epoch_start = time.time()

        avg_loss, KLD_loss, RKLD_loss, WKLD_loss, JSD_loss, count = 0, 0, 0, 0, 0, 0

        for batch_data in tqdm(origin_loader):

            batch_image = batch_data['images']
            g_batch_image = batch_data['g_images']

            x_1 = batch_image.to(device)
            x_2 = g_batch_image.to(device)

            optimizer.zero_grad()

            x1_out = net(x_1)
            x2_out = net(x_2)

            loss = IID_loss(x1_out, x2_out)

            # kl1 = KL_divergence(x1_out, x2_out)
            # kl2 = KL_divergence(x2_out, x1_out)

            # KLD_loss += kl1
            # RKLD_loss += kl2
            # WKLD_loss += (kl1 + kl2)

            # js = JS_divergence(x1_out, x2_out)
            # JSD_loss += js

            avg_loss += loss.item()

            loss.backward()
            optimizer.step()

        # KLD_loss = KLD_loss.to('cpu').detach().numpy().copy()
        # RKLD_loss = RKLD_loss.to('cpu').detach().numpy().copy()
        # WKLD_loss = WKLD_loss.to('cpu').detach().numpy().copy()
        # JSD_loss = JSD_loss.to('cpu').detach().numpy().copy()

        print('EPOCH: {}, epoch_loss: {}'.format(
            epoch+1, avg_loss))
        
        elapsed_time = time.time() - epoch_start
        print('execution time:{0}'.format(elapsed_time) + '[sec]')

        train_loss_value.append(
            [avg_loss, KLD_loss, RKLD_loss, WKLD_loss, JSD_loss]
        )

        if os.path.exists(net_path) == False:
            os.makedirs(net_path)

        if avg_loss < min_avg_loss:

            min_avg_loss = avg_loss
            bestLossEpoch = epoch

            print('update min_loss = ', min_avg_loss)

            if os.path.exists(net_path) == False:
                os.makedirs(net_path)

            checkpoint = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state(),
            }

            torch.save(checkpoint, os.path.join(net_path,'checkpoint_best.bin'))

            print('new best model saved!')

            early_count = 0

        else:
            early_count += 1

            if early_count == config.TRAIN.LR_STEP:
                scheduler.step()
                print('define new lr param')

            if early_count == config.TRAIN.EARLY_STOP:
                print('early_stop')
                print('stop epoch : ', epoch+1)
                print('-'*50)
                break

        print('-'*50)

    elapsed_time = time.time() - start

    print('execution time: {0}'.format(elapsed_time) + '[sec]')

    if os.path.exists(net_path) == False:
        os.makedirs(net_path)

    checkpoint = {
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        "cuda_random": torch.cuda.get_rng_state(),
    }

    torch.save(checkpoint, net_path+'checkpoint_last.bin')

    loss_List = ['Loss', 'KLD', 'RKLD', 'W-KLD', 'JSD']

    for i in loss_List:
        make_graph('Epochs', i, loss_List, train_loss_value, net_path, no)
    print('Complete make graph\n')

    hist_df = pd.DataFrame(train_loss_value)
    hist_df.to_csv(net_path + 'loss_' + str(no) + '.csv')


if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)

            



