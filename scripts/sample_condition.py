from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.ddim import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator, normalize_torch
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
from functools import partial
from ldm.data.lsun import LSUNBedroomsValidation
import numpy as np

import os
import torch
import torchvision.transforms as transforms
import argparse
import yaml
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from torchmetrics import FID, SSIM, KID, LPIPS
from skimage.metrics import peak_signal_noise_ratio as psnr
#from torchmetrics.image import PeakSignalNoiseRatio

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_model_from_config(config, ckpt, train=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)
    
    model.cuda()

    if train:
      model.train()
    else:
      model.eval()

    return model


def get_model(args):
    model_config = OmegaConf.load(args.diffusion_config)
    model = instantiate_from_config(model_config.model)

    return model

#def get_model(model_type = None):
#    if model_type is None:
#      config = OmegaConf.load("configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
#      #config = OmegaConf.load("configs/latent-diffusion/celebahq-ldm-vq-4.yaml")
#      #model = load_model_from_config(config, "models/ldm/ffhq/ffhq_model.ckpt")
#      model = load_model_from_config(config, "models/ldm/lsun_bedrooms/model.ckpt")
#      #model = load_model_from_config(config, "models/ldm/celeba256/model.ckpt") 
#    return model


parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", type=str)
parser.add_argument('--model_config', default="models/ldm/ffhq/model.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/gaussian_deblur_config.yaml", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)

args = parser.parse_args()


# Load configurations
task_config = load_yaml(args.task_config)

# Device setting
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  

# Loading model
model = get_model(args)
#model = get_model()
sampler = DDIMSampler(model) # Sampling using DDIM

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=None,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[3, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta)

# Working directory
out_path = os.path.join(args.save_dir)
os.makedirs(out_path, exist_ok=True)
for img_dir in ['input', 'recon', 'progress', 'label']:
    os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
#dataset = LSUNBedroomsValidation(txt_file='./data/lsun/bedrooms_val_100_images.txt')
dataset = LSUNBedroomsValidation()

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
  mask_gen = mask_generator(**measure_config['mask_opt'])

# Temporary stuff
psnr_all = []
lpips_all = []
ssim_all = []

# Do inference
for i, ref_img in enumerate(loader):

    print(f"Inference for image {i}")
    fname = str(i).zfill(3)
    ref_img = ref_img.to(device)

    # Exception) In case of inpainting
    if measure_config['operator'] ['name'] == 'inpainting':
      mask = mask_gen(ref_img)
      mask = mask[:, 0, :, :].unsqueeze(dim=0)
      operator_fn = partial(operator.forward, mask=mask)
      measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
      sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

      # Forward measurement model
      y = operator_fn(ref_img)
      y_n = noiser(y)

    else:
      y = operator.forward(ref_img)
      y_n = noiser(y).to(device)

    # Sampling
    samples_ddim, _ = sample_fn(measurement=y_n, true_img=ref_img)
    
    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

    # Post-processing samples
    label = clear_color(y_n)
    reconstructed = clear_color(x_samples_ddim)
    true = clear_color(ref_img)

    # Saving images
    plt.imsave(os.path.join(out_path, fname+'_true.png'), true)
    plt.imsave(os.path.join(out_path, fname+'_label.png'), label)
    plt.imsave(os.path.join(out_path, fname+'_reconstr.png'), reconstructed)

    # Getting metrics
    ssim = SSIM()
    ssim_cur = ssim(x_samples_ddim, ref_img)

    lpips = LPIPS(net_type='vgg')
    lpips_cur = lpips(normalize_torch(x_samples_ddim).cpu(), normalize_torch(ref_img).cpu())

    psnr_cur = psnr(true, reconstructed)
    print('LPIPS:', lpips_cur)
    print('PSNR:', psnr_cur)
    print('SSIM:', ssim_cur)

    psnr_all.append(psnr_cur)
    ssim_all.append(ssim_cur.item())
    lpips_all.append(lpips_cur.item())

    print('Running Average PSNR:', np.mean(psnr_all))
    print('Running Average SSIM:', np.mean(ssim_all))
    print('Running Average LPIPS:', np.mean(lpips_all))

    print('Running STD PSNR:', np.std(psnr_all))
    print('Running STD SSIM:', np.std(ssim_all))
    print('Running STD LPIPS:', np.std(lpips_all))


print('Final Average PSNR:', np.mean(psnr_all))
print('Final Average SSIM:', np.mean(ssim_all))
print('Final Average LPIPS:', np.mean(lpips_all))

print('Final STD PSNR:', np.std(psnr_all))
print('Final STD SSIM:', np.std(ssim_all))
print('Final STD LPIPS:', np.std(lpips_all))