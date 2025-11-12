from functools import partial
import os
import argparse
import yaml

import torch
import matplotlib.pyplot as plt
import csv
import os
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.fgps_model import FGPSModel
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from guided_diffusion.data import get_data
from guided_diffusion.params import parse_args as p
from torchvision.transforms import transforms, InterpolationMode, Resize
from pdb import set_trace
import pandas as pd
from PIL import Image

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='results/')
    args = parser.parse_args()

    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']

    if cond_config['method'] == 'fgps':
        fgps_model= FGPSModel(device)
        fgps_model = fgps_model.to(device)
        fgps_model.eval()
        cond_model = fgps_model
    else:
        cond_model = None
    out_path = os.path.join(args.save_dir, f"{measure_config['operator']['name']}_{task_config['conditioning']['method']}")
  
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, cond_model=cond_model, T=diffusion_config['steps'], **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    Resize((256,256), interpolation=InterpolationMode.BICUBIC),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    dataset = get_dataset(**data_config, transforms=transform)#, mode="val", dataset_size=1000)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    for index, image in enumerate(loader):
        fname = str(index).zfill(5) + '.png'
        if data_config['name'] == 'imagenet':
            image, label = image
            label = label.to(device)
        else:
            label = None
        clean_images = image.to(device)

        y = operator.forward(clean_images)
        y_n = noiser(y)
        plotted_yn = clear_color(y_n)
        cond_method.init_mags(y_n)
        # Sampling
        x_start = torch.randn(clean_images.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path, label=label)
        if len(plotted_yn.shape) == 2:
            plt.imsave(os.path.join(out_path, 'input', fname), plotted_yn, cmap='gray')
        else:
            plt.imsave(os.path.join(out_path, 'input', fname), plotted_yn)
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(clean_images))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
        
        psnr_cur = psnr(clear_color(clean_images), clear_color(sample))
        print('PSNR:', psnr_cur)

if __name__ == '__main__':
    main()
