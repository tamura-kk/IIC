from argparse import ArgumentParser
import torchvision
from torchvision import transforms
from config import get_config
import sys
import torch
import random
import numpy as np
import time
from glob import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
from models.ModelDefine import NET
from argparse import ArgumentParser
import multiprocessing
import argparse
import shutil
import pandas as pd


# def get_args():
#     parser = ArgumentParser(
#         description='This is sample argparse script')
#     parser.add_argument('-n', '--no', default=0,
#                         type=int, help='This is name.')
#     parser.add_argument('-c', '--out_c', default=0,
#                         type=int, help='This is name.')
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

def resize_img(config):
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


def main(config):
    
    start = time.time()

    no = config.MODEL.NUMBER
    out_c = config.DATA.OUTPUT_CHANNEL
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE
    print('out classes ----> ', out_c)
    print('input_size ----> ', config.DATA.IMAGE_SIZE)

    if out_c > 0:
        net = NET(config)
    else:
        print('model is not Defined')
        sys.exit()

    print('use number of gpu ----> ', torch.cuda.device_count(),)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('device_name ----> ', device)
    print('experiment_number ----> ', no)
    
    base_path =  os.path.join('result', str(data_type),str(model_name),str(no))
    net_path =  os.path.join(base_path,'train', 'class_model')

    checkpoint = torch.load(os.path.join(net_path,'checkpoint_best.bin'))
    
    result_path = os.path.join(base_path,'test')
    
    os.makedirs(result_path, exist_ok=True)
    
    shutil.copyfile('config.py',os.path.join(result_path,'config_{}.py'.format(no)))

    f = open(os.path.join(result_path ,'conditions.txt'), 'a')

    datalist = ['Experiment No. : {}\n'.format(no),
                'Image size : {}\n'.format(config.DATA.IMAGE_SIZE),
                'Class : {}\n'.format(out_c),
                'Model name : {}\n'.format(config.MODEL.TYPE),
                'data_path : {}\n'.format(config.DATA.TRAIN_DATA_PATH),
                'data_type :{}\n'.format(config.DATA.TYPE),
                '-'*50, '\n' ]
    
    f.writelines(datalist)
    f.close()

    net.to(device)

    net.load_state_dict(checkpoint['model'])
    random.setstate(checkpoint['random'])
    np.random.set_state(checkpoint['np_random'])
    torch.set_rng_state(checkpoint['torch'])
    torch.random.set_rng_state(checkpoint['torch_random'])
    torch.cuda.set_rng_state(checkpoint['cuda_random'])

    test_trans = resize_img(config)

    net.eval()
    path_list = glob(config.DATA.TEST_DATA_PATH)

    print('number of images', len(path_list))
    columns = ['patient', 'image_name', 'type', 'class']
    result = pd.DataFrame(columns=columns)
    result_extraction = pd.DataFrame(columns=columns)

    count = 0
    outVecs = []
    pathVecs = []

    for b_p in tqdm(path_list):

        image = Image.open(b_p)
        image = test_trans(image)
        image = image.unsqueeze_(0)

        image = image.clone().detach().float()
        image = image.to(device)

        if config.DATA.COLOR != 'colornorm':
            image = image / 255.0

        
        out = net(image)
        _, out_id = out.max(1)

        human_id = os.path.basename(os.path.dirname(b_p))
        diabetes = os.path.basename(os.path.dirname(os.path.dirname(b_p)))
        basename = os.path.basename(b_p)

        data = [[human_id, basename, diabetes, int(out_id)]]
        tmp = pd.DataFrame(data=data, columns=columns)
        result = pd.concat([result, tmp], ignore_index=True, axis=0)

        if torch.max(out) >= 0.99:
            count += 1
            with torch.no_grad():
                outvec = net.forward_feature(image)

            outvec = outvec.to('cpu').detach().numpy().copy()

            outVecs.append(outvec[0])
            pathVecs.append(b_p)
            result_extraction = pd.concat([result_extraction, tmp], ignore_index=True, axis=0)

    print('Number of images for which threshold >= 0.99', count, 'sheets')
    
    result.to_csv(os.path.join(result_path ,'IIC_result_{}.csv'.format(no)), index=False)
    result_extraction.to_csv(os.path.join(result_path ,'IIC_result_extraction_{}.csv'.format(no)), index=False)

    os.makedirs(os.path.join(result_path ,'clustering_extraction'), exist_ok=True)

    outVecs = np.array(outVecs)
    pathVecs = np.array(pathVecs)

    np.save(os.path.join(result_path ,'clustering_extraction', 'outVecs_{}'.format(no)), outVecs)
    np.save(os.path.join(result_path ,'clustering_extraction', 'path_list_{}'.format(no)), pathVecs)

    print('End of make npy files')

    elapsed_time = time.time() - start

    print('excution_time : {0}'.format(elapsed_time) + '[sec]')


if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)



