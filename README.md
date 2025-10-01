# Frequency Guided Posterior Sampling for Diffusion-Based Image Restoration (ICCV 2025)
### 📝 [Paper](https://arxiv.org/abs/2411.15295)

## Abstract

   Image restoration aims to recover high-quality images from degraded observations. When the degradation process is known, the recovery problem can be formulated as an inverse problem, and in a Bayesian context, the goal is to sample a clean reconstruction given the degraded observation. Recently, modern pretrained diffusion models have been used for image restoration by modifying their sampling procedure to account for the degradation process. However, these methods often rely on certain approximations that can lead to significant errors and compromised sample quality. In this paper, we propose a simple modification to existing diffusion-based restoration methods that exploits the frequency structure of the reverse diffusion process. Specifically, our approach, denoted as Frequency Guided Posterior Sampling (FGPS), introduces a time-varying low-pass filter in the frequency domain of the measurements, progressively incorporating higher frequencies during the restoration process. We provide the first rigorous analysis of the approximation error of FGPS for linear inverse problems under distributional assumptions on the space of natural images, demonstrating cases where previous works can fail dramatically. On real-world data, we develop an adaptive curriculum for our method's frequency schedule based on the underlying data distribution. FGPS significantly improves performance on challenging image restoration tasks including motion deblurring and image dehazing.

## Prerequisites 

- Python 3.9 
- Pytorch 2.6.0
- CUDA 12.5

For lower versions of Pytorch, the appropriate version of Python and CUDA can likely be used, although the code was tested only with the above configuration.

<br /> 

## Getting Started

### 1) Download pretrained checkpoints

Please refer to the pretrained FFHQ and Imagenet checkpoints from [DPS codebase](https://github.com/DPS2022/diffusion-posterior-sampling) and place it in checkpoints/ directory.

This repository builds on top the implementation provided in [DPS codebase](https://github.com/DPS2022/diffusion-posterior-sampling).

<br />

### 2) Install dependencies 

```
pip install -r requirements.txt
git clone https://github.com/LeviBorodenko/motionblur motionblur
```

### 3) Run inference


```
python fgps_sample_condition.py \
    --model_config configs/model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config configs/{TASK_CONFIG}.yaml 
    --save_dir results
```

Possible task configurations are given in configs/ directory. 

## Citation

If you find our work interesting, please consider citing

```
@inproceedings{thaker2024frequency,
  title={Frequency-guided posterior sampling for diffusion-based image restoration},
  author={Thaker, Darshan and Goyal, Abhishek and Vidal, Ren{\'e}},
  journal={International Conference on Computer Vision 2025},
  year={2025}
}
```