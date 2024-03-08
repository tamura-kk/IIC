########
# Perform 512 dimensional feature extraction using the model trained in IIC and dimensionality reduction using a saved .npy file
# Create a scatter plot with coloring for each class
# execute with the following command
# python tsne_IICmodel.py -n 0 -p 5 -l 100 -d S
########


import numpy as np
from glob import glob
import random
from PIL import Image
import time
import datetime
import os
import time
import shutil
import sys
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
# import umap
import argparse
from config import get_config
import multiprocessing
import warnings
warnings.simplefilter('ignore')



# def get_args():
#     parser = argparse.ArgumentParser(
#         description='This is sample argparse script')
#     parser.add_argument('-n', '--no', default=0,
#                         type=int, help='This is name.')
#     parser.add_argument('-p', '--perplexity', default=5,
#                         type=int, help='This is name.')
#     parser.add_argument('-l', '--lim', default=100,
#                         type=int, help='This is name.')
#     parser.add_argument('-d', '--disc', default='D',
#                         type=str, help='This is disc.')
#     return parser.parse_args()

def parse_option():
    parser = argparse.ArgumentParser('tsne script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config



def main(config):

    no = str(config.MODEL.NUMBER)
    perp_list = config.TSNE.PERPLEXITY_LIST
    lim = config.TSNE.LIM
    # load .csv file

    # Specify figure size
    fig = plt.figure(figsize=(6, 6))   # width6, height6
    # How many classes of vectors to use
    label = 10  # 3, 5, 10
    start = time.time()
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE

    base_path = os.path.join('result', data_type, model_name, no,'test')


    # dataframeとpath_listとout_vecsを読み込む
    df = pd.read_csv(
        os.path.join(base_path,'IIC_result_extraction_{}.csv'.format(no)), index_col=0
    )
    out_vecs = np.load(os.path.join(base_path,'clustering_extraction', 'outVecs_{}.npy'.format(no))
    )
    path_list = np.load(os.path.join(base_path,'clustering_extraction', 'path_list_{}.npy'.format(no))
    )
    #     disc+':OmoteShogo/result/IIC/{}/IICresult_chuusyutu_{}.csv'.format(no,no), index_col=0)
    # out_vecs = np.load(disc+':OmoteShogo/result/IIC/{}/clustering_chuusyutu/outVecs_{}.npy'.format(no, no))
    # path_list = np.load(disc+":OmoteShogo/result/IIC/{}/clustering_chuusyutu/path_list_{}.npy".format(no, no))

    # Specify the color of each cluster
    colormap = ['r', 'g', 'b', 'k', 'y', 'c', 'm',
                'lime', 'violet', 'coral', 'slategray']
    colorlist = []
    labellist = []
    for b_p in path_list:
        img_name = os.path.basename(b_p)
        col = df[df['image_name'] == img_name]
        tmp = col['class'].values
        colorlist.append(colormap[tmp[0]])
        labellist.append(str(tmp[0]))
    # Execution of t-SNE
    for perp in perp_list:
        tsne = TSNE(n_components=2, random_state=0, perplexity=perp, n_iter=1000)
        # uma = umap.UMAP(n_components=2, random_state=0, n_neighbors=n_nei, min_dist=0.1, metric='correlation')
        X_embedded = tsne.fit_transform(out_vecs)   # numpy.ndarray

        # Extract x and y coordinates
        x = X_embedded[:, 0]
        y = X_embedded[:, 1]

        # Drawing Graphs
        plt.scatter(x, y, s=10, alpha=0.8, c=colorlist)
        fig.legend()
        plt.grid()
        plt.title('class'+str(label))
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        # Save Graph
        print('-'*10 + 'saving Graph...' + '-'*10)
        result_path = os.path.join(base_path,'t-SNE_graph'.format(no))
        os.makedirs(result_path, exist_ok=True)
        fig.savefig(os.path.join(result_path,'tsne_IICmodel_n{}_p{}.png'.format(no, perp)))
        elapsed_time = time.time() - start
        print('Took {:.0f} seconds to execute.'.format(elapsed_time))


if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)
