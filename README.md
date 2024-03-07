# Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency (ICLR 2024)

![example](https://github.com/soominkwon/resample/blob/main/figures/resample_ex.png)

## Getting Started

### 1) Clone the repository

```
git clone https://github.com/soominkwon/resample

cd resample
```

<br />

### 2) Download pretrained checkpoints (autoencoders and model)

```
mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

mkdir models/first_stage_models
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models
unzip models/first_stage_models/vq-f4.zip -d ./models/first_stage_models
```

<br />

### 3) Set environment

We use the external codes for motion-blurring and non-linear deblurring following the DPS codebase.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies

```
conda create -n ldm python=3.8

conda activate ldm

pip install -r requirements.txt TODO: Add this
```

<br />

### 4) Inference

```
python3 sample_condition.py
```

The code is currently configured to do inference on FFHQ. You can download the corresponding models from https://github.com/CompVis/latent-diffusion/tree/main and modify the checkpoint paths for other datasets and models.


<br />

## Task Configurations

```
# Linear inverse problems
- configs/tasks/super_resolution_config.yaml
- configs/tasks/gaussian_deblur_config.yaml
- configs/tasks/motion_deblur_config.yaml
- configs/tasks/inpainting_config.yaml

# Non-linear inverse problems
- configs/tasks/nonlinear_deblur_config.yaml
```

## Citation
If you find our work interesting, please consider citing

```
@inproceedings{
song2024solving,
title={Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency},
author={Bowen Song and Soo Min Kwon and Zecheng Zhang and Xinyu Hu and Qing Qu and Liyue Shen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=j8hdRqOUhN}
}
```

