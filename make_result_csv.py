from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys
import argparse
from config import get_config
import multiprocessing
import os


# def get_args():
#     parser = ArgumentParser(
#         description = 'This is sample argparse script'
#     )
#     parser.add_argument('-n', '--no', default=0,
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

def main(config):

    no = config.MODEL.NUMBER
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE
    
    result_path = os.path.join('result', str(data_type), str(model_name), str(no), 'test')

    result = pd.read_csv(os.path.join(result_path, 'IIC_result_{}.csv'.format(no)))
    
    outClass = result['class'].unique()
    outClass = np.sort(outClass)

    print(outClass)
    
    result = pd.get_dummies(result, columns=['class'])

    name_list = result['patient'].unique()
    
    type_list = result['type'].unique()

    class_list = ['patient', 'type']
    
    for c in outClass:
        class_list.append('class' + str(c))
    
    count_result = pd.DataFrame()
    
    for name in name_list:
        tmp = result[result['patient'] == name]
        
        tmp = tmp.drop(['image_name'], axis=1)
        
        numImages = len(tmp)
        
        tmp2 = tmp.sum(axis=0)
        
        tmp2['patient'] = name
        
        if 'not' in tmp2['type']:
            tmp2['type'] = 'not_diabetes'
        else:
            tmp2['type'] = 'diabetes'

        if tmp2['type'] == 'diabetes':
            tmp2['type'] = 1
        else:
            tmp2['type'] = 0
        
        for c in outClass:

            tmp2['class_' + str(c)] = tmp2['class_' + str(c)] / numImages
            
        tmp2 = pd.DataFrame(tmp2).transpose()
         
        count_result = pd.concat([count_result, tmp2], ignore_index=True)

    count_result = count_result.rename(columns={'patient': 'name'})

    for c in outClass:
        count_result = count_result.rename(columns={'class_' + str(c): c})

    print('-'*50)

    print(count_result.head(5))

    count_result.to_csv(os.path.join(result_path, 'CountResult_{}.csv'.format(no)))

if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)