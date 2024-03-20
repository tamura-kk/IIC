import pandas as pd
import argparse
from config import get_config
import multiprocessing
import os
import pandas as pd
import shutil
import io

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

def create_path(row):
    return os.path.join(row['patient'], row['image_name'])

def main(config):

    no = config.MODEL.NUMBER
    out_c = config.DATA.OUTPUT_CHANNEL
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE
    
    result_path =  os.path.join('result', str(data_type),  str(model_name), str(no), 'test')

    df = pd.read_csv(os.path.join(result_path, 'IIC_result_{}.csv'.format(no)))

    #csvファイルから各画像のパスのdfを作成
    df['image_path'] =  'dataset/' + data_type + '/' +  df['type'] + '/' + df['patient'] + '/' + df['image_name']

    for cls in range(out_c):
        #特定のクラスのみのdfを抽出
        class_indices = df[df['class'] == cls].index
        for index in class_indices:
            #元のデータのパス
            img_path = df.at[index, 'image_path']

            base_name = os.path.basename(img_path)
            human_id = os.path.basename(os.path.dirname(img_path))
            type = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

            #クラスタリング結果によるデータのパス
            # cluster_path = os.path.join(result_path, 'cluster_data', str(cls) ,type, human_id, base_name)
            cluster_path = os.path.join(result_path, 'cluster_data', str(cls) ,base_name)

            #フォルダー作成
            os.makedirs(os.path.dirname(cluster_path), exist_ok=True)
            shutil.copy(img_path, cluster_path)

if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)
