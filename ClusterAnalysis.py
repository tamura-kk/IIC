from argparse import ArgumentParser
import pandas as pd
import argparse
from config import get_config
import multiprocessing
import os
import pandas as pd


# def get_args():
#     parser = ArgumentParser(
#         description = 'This is sample argument script')
#     parser.add_argument('-n', '--no',
#                         default=0, type=int)
#     parser.add_argument('-o', '--out_c',
#                         default=0, type=int)
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
    out_c = config.DATA.OUTPUT_CHANNEL
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE
    
    result_path =  os.path.join('result', str(data_type),  str(model_name), str(no), 'test')

    df = pd.read_csv(os.path.join(result_path, 'IIC_result_{}.csv'.format(no)), index_col=0)

    tmp = []

    for cls in range(out_c):
        label = str(cls)

        count1 = ((df['type'] == 'diabetes') & (df['class'] == cls)).sum()
        count0 = ((df['type'] == 'not_diabetes') & (df['class'] == cls)).sum()

        s = count1 + count0

        tmp.append([label, count1, count0, s])

    df_sum = pd.DataFrame(
        tmp, columns=['labels', 'diabetes', 'not_diabetes', 'diabetes + not_diabetes']
    )

    df_sum.to_csv(os.path.join(result_path, 'ClusterAnalysis_{}.csv'.format(no)))

if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)
