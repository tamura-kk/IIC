import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------

_C.DATA = CN()
_C.DATA.TRAIN_DATA_PATH = 'dataset/*/*/*.jpg'
_C.DATA.TEST_DATA_PATH = 'dataset/*/*/*.jpg'
_C.DATA.IMAGE_SIZE = 224
_C.DATA.BATCH_SIZE = 128
_C.DATA.OUTPUT_CHANNEL = 10
_C.DATA.COLOR = 'gray'
_C.DATA.NUM_WORKERS = 1
_C.DATA.TYPE = None

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.TYPE = 'swin'
_C.MODEL.NUMBER = 0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.NUM_CLASS = 10
_C.MODEL.CHECKPOINT = False

#Swin_Transformer Parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

#ResNet Parameters
_C.MODEL.RESNET50 = CN()
_C.MODEL.RESNET50.LAYER_LIST = [3, 4, 5, 6]
_C.MODEL.RESNET50.NUM_CHANNELS = 3
_C.MODEL.RESNET50.IN_CHANNELS = 64
_C.MODEL.RESNET50.OUT_CHANNELS = 64

#MedVit　Parameters
_C.MODEL.MEDVIT = CN()
_C.MODEL.MEDVIT.STEM_CHS =  [64, 32, 64]
_C.MODEL.MEDVIT.DEPTHS = [3, 4, 10, 3]
_C.MODEL.MEDVIT.PATH_DROPOUT = 0.1
_C.MODEL.MEDVIT.ATTN_DROP =  0
_C.MODEL.MEDVIT.DROP = 0
_C.MODEL.MEDVIT.STRIDES = [1, 2, 2, 2]
_C.MODEL.MEDVIT.SR_RATIOS = [8, 4, 2, 1]
_C.MODEL.MEDVIT.HEAD_DIM = 32
_C.MODEL.MEDVIT.MIN_BLOCK_RATIO = 0.75


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 300
_C.TRAIN.LR = 5e-4
_C.TRAIN.EARLY_STOP = 40
_C.TRAIN.LR_STEP = 5
_C.TRAIN.LOAD_MODEL_TYPE = 'best'
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.OVER_CLUSTERING = False

_C.FUSED_WINDOW_PROCESS = False

# -----------------------------------------------------------------------------
# TSNE settings
# -----------------------------------------------------------------------------

_C.TSNE = CN()
_C.TSNE.PERPLEXITY_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
_C.TSNE.LIM = 100

# -----------------------------------------------------------------------------
# CAM settings
# -----------------------------------------------------------------------------

_C.CAM = CN()
_C.CAM.DEVICE ='cuda'
_C.CAM.METHOD = 'scorecam'
_C.CAM.IMAGE_PATH = None
_C.CAM.OUT_PATH = None




def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
        
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config