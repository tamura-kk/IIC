import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random
from config import get_config
from models.ModelDefine import NET
import multiprocessing
import torchvision
from torchvision import transforms
from torchsummary import summary
from PIL import Image


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
            transforms.PILToTensor(),  
            transforms.Normalize(mean=[0.728, 0.514, 0.708],  
                             std=[0.380, 0.446, 0.471])
        ])
    return transform

def main(config):
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    method = config.CAM.METHOD
    img_path = config.CAM.IMAGE_PATH
    out_path = config.CAM.OUT_PATH
    device = config.CAM.DEVICE
    no = config.MODEL.NUMBER
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE

    base_path =  os.path.join('result', str(data_type),str(model_name),str(no))
    net_path =  os.path.join(base_path,'train', 'class_model')
    checkpoint = torch.load(os.path.join(net_path,'checkpoint_best.bin'))
    net = NET(config)

    print('method :',method)
    print('device :', device)
    print('image_path :', img_path)
    print('data_type :', data_type)
    print('model :', model_name)

    net.to(device)


    net.load_state_dict(checkpoint['model'])
    random.setstate(checkpoint['random'])
    np.random.set_state(checkpoint['np_random'])
    torch.set_rng_state(checkpoint['torch'])
    torch.random.set_rng_state(checkpoint['torch_random'])
    torch.cuda.set_rng_state(checkpoint['cuda_random'])
    net.eval()

    # model = models.resnet50(pretrained=True).to(torch.device(device)).eval()


    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    summary(net, (3, config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE))
    target_layers = [net.model.layer4]

    test_trans = resize_img(config)

    image = Image.open(img_path)
    image = test_trans(image)
    image = image.unsqueeze_(0)

    image = image.clone().detach().float()
    image = image.to(device)

    if config.DATA.COLOR != 'colornorm':
        image = image / 255.0
    
    out = net(image)
    print(out)
    _, out_id = out.max(1)
    print(out_id)


    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[method]
    with cam_algorithm(model=net,
                       target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=image,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=net, device=device)
    gb = gb_model(image, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    os.makedirs(out_path, exist_ok=True)

    cam_output_path = os.path.join(out_path, f'{method}_cam.jpg')
    gb_output_path = os.path.join(out_path, f'{method}_gb.jpg')
    cam_gb_output_path = os.path.join(out_path, f'{method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)

if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)

