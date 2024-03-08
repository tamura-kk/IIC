from .SwinTransformer import SwinTransformer
from .resnet50 import *
from .MedViT import MedViT
import torch.nn as nn

def build_model(config):
    model_type = config.MODEL.TYPE
    
    layernorm = nn.LayerNorm

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMAGE_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'medvit':
        model = MedViT(stem_chs=config.MODEL.MEDVIT.STEM_CHS, 
                       depths=config.MODEL.MEDVIT.DEPTHS, 
                       path_dropout=config.MODEL.MEDVIT.PATH_DROPOUT, 
                       attn_drop=config.MODEL.MEDVIT.ATTN_DROP, 
                       drop=config.MODEL.MEDVIT.DROP, 
                       num_classes=config.MODEL.NUM_CLASS,
                       strides=config.MODEL.MEDVIT.STRIDES, 
                       sr_ratios=config.MODEL.MEDVIT.SR_RATIOS, 
                       head_dim=config.MODEL.MEDVIT.HEAD_DIM, 
                       mix_block_ratio=config.MODEL.MEDVIT.MIN_BLOCK_RATIO,
                       use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )       
        
    elif model_type == 'resnet50':
        model = ResNet(
            ResBlock = Bottleneck,
            layer_list = config.MODEL.RESNET50.LAYER_LIST,
            num_classes = config.MODEL.NUM_CLASS,
            num_channels = config.MODEL.RESNET50.NUM_CHANNELS
        )
        
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    return model
