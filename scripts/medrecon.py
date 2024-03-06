


###before run this, run bash scripts/download_models.sh and bash scripts/download_first_stages.sh
###remember to sample uniformly (if every data point (t, x, z, etc) is distributed evenly).


#from taming.models import vqgan
from ldm.models.diffusion.ddim import DDIMSampler
#@title loading utils
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from einops import rearrange, repeat
from utils import *
import PIL
import argparse
import scripts.utils as utils



def load_model_from_config(config, ckpt, train=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    model.cuda()
    # model.train()
    if train:
      model.train()
    else:
      model.eval()
    return model


def get_model(model_type = None):
    if model_type is None:
      config = OmegaConf.load("configs/latent-diffusion/celebahq-ldm-vq-4.yaml")
      model = load_model_from_config(config, "models/ldm/celeba256/model.ckpt") 
    elif model_type == "celeb":
      config = OmegaConf.load("configs/latent-diffusion/celebahq-ldm-vq-4.yaml") 
      model = load_model_from_config(config, "models/ldm/celeba256/model.ckpt")
    else:
      model = None
    return model



def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def load_img2(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((256,256), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.



parser = argparse.ArgumentParser()
parser.add_argument('--sampling', type=int, default=-1, help='sample image')
parser.add_argument('--finetune', type=int, default=-1, help="fine tune")
parser.add_argument('--ctrecon', default=False, action='store_true', help="load pretrained model weights")
# Load experiment setting
opts = parser.parse_args()
samplesteps, finetunesteps, ctreconflag = opts.sampling, opts.finetune, opts.ctrecon



model = get_model()
print(model)
print(type(model))
print(model.num_timesteps) ###=1000
n_samples_per_class = 6
device = torch.device("cuda")


"""

if finetunesteps > -1:
  model.train()
  init_image = load_img("/content/drive/MyDrive/stable-diffusion/ldct/test_ct_img0.png").to(device)
  init_image = repeat(init_image, '1 ... -> b ...', b=n_samples_per_class)
  t = repeat(torch.tensor([5]), '1 -> b', b=6)
  t = t.to(device).long()
  model.learning_rate = 1e-5
  num_timesteps = 1000
  opt = model.configure_optimizers()
  print(type(model))
  init_image = load_img("/content/drive/MyDrive/stable-diffusion/ldct/test_ct_img" + str(0) + ".png").to(device)

  init_image = load_img("/content/drive/MyDrive/stable-diffusion/ldct/test_recon.png").to(device)
  init_image = load_img2("/content/drive/MyDrive/stable-diffusion/maolilan.png").to(device)
  init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
  losses = []
  for i in range(20000):
    img_index = np.random.randint(10, 1000)
    init_image = load_img("/content/drive/MyDrive/stable-diffusion/test_ct_img" + str(img_index) + ".png")
    for j in range(5):
      img_index = np.random.randint(10, 1000)
      new_image = load_img("/content/drive/MyDrive/stable-diffusion/test_ct_img" + str(img_index) + ".png")
      init_image = torch.vstack((init_image, new_image))
    init_image = init_image.to(device)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image)) ###is this deterministic TODO need to check
    
    batch_loss = None 
    t = torch.randint(0, model.num_timesteps, ((init_latent.shape)[0],), device=device).long()
    loss, loss_dict = model.p_losses(init_latent, None, t)
    if batch_loss is None:
      batch_loss = loss
    else:
      batch_loss += loss
    losses.append(batch_loss.detach().cpu().numpy())
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    print(i, "th iteration")
  # for j in range(1):
  #   t_random = np.random.randint(20)
  #   t = repeat(torch.tensor([t_random]), '1 -> b', b=2)
  #   t = t.to(device).long()
  #   opt.zero_grad()
  #   # xc = torch.tensor(n_samples_per_class*[0])
  #   # c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
  #   # np.save("conditioncode" + str(j) + ".npy",c.detach().cpu().numpy())
  #   # loss, loss_dict = model.p_losses(init_latent, c, t). ###only for conditioned models
  #   loss, loss_dict = model.p_losses(init_latent, None, t)
  #   print(loss.detach().cpu().numpy(), "loss")
  #   loss.backward()
  #   opt.step()

  np.save("losses_1000i_10t.npy", np.array(losses))


  model.eval()
  samples, intermediates = model.sample(cond=None, batch_size=6,
                                                 return_intermediates=True)

  x_samples_ddpm = model.decode_first_stage(samples)

  np.save("sampled_imgs_1000i_10t.npy", x_samples_ddpm.cpu().numpy())
  np.save("DDPM_samples_1000i_10t.npy", samples.cpu().numpy())
# np.save("Latent_intermediates.npy", intermediates.cpu().numpy())                                               



if ctreconflag:
  init_image = load_img("/content/drive/MyDrive/stable-diffusion/ldct/test_recon.png").to(device)
  init_image = repeat(init_image, '1 ... -> b ...', b=n_samples_per_class)
  thetas = torch.tensor(np.linspace(0,np.pi,50, endpoint=False))
  projections = ct_parallel_project_2d_batch(init_image.permute(0,2,3,1), thetas)
  # projections = ct_parallel_project_2d_batch(init_latent.permute(0,2,3,1), thetas)
  # projections = init_latent
  print(init_image.shape)
  print(projections.shape)
  np.save("../projections.npy", projections.detach().cpu().numpy())
  model.eval()
  sampler = DDIMSampler(model)

#####inverse problem solving
  ddim_steps = 200
  all_samples = list()
  n_samples_per_class = 1

  ddim_steps = 1000
  ddim_eta = 0.0
  scale = 1.0   # for unconditional guidance
  ####set ddim_use_original_steps = True for sampling every timesteps
  samples_ddim, intermediates = sampler.inverse_sample(S=ddim_steps,
                                             conditioning=None,
                                             ddim_use_original_steps = True,
                                             batch_size=n_samples_per_class,
                                             shape=[3, 64, 64],
                                             y = projections,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=None, 
                                             eta=ddim_eta)
  x_samples_ddim = model.decode_first_stage(samples_ddim.detach())
  x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

  inter_recon = []
  for latent_vec in intermediates['pred_x0']:
    intermediates_ddim = model.decode_first_stage(latent_vec.detach())
    intermediates_ddim = torch.clamp((intermediates_ddim+1)/2, min=0.0, max=1.0)
    inter_recon.append(intermediates_ddim.cpu().numpy())


  inter_recon2 = []
  for latent_vec in intermediates['x_inter']:
    intermediates_ddim = model.decode_first_stage(latent_vec.detach())
    intermediates_ddim = torch.clamp((intermediates_ddim+1)/2, min=0.0, max=1.0)
    inter_recon2.append(intermediates_ddim.cpu().numpy())
  np.save('inversesampletest392023_50.npy', x_samples_ddim.cpu().numpy())
  np.save('intermediatestest392023_50.npy', np.array(inter_recon))
  np.save('intermediatestest2392023_50.npy', np.array(inter_recon2))

    



"""
import matplotlib.pyplot as plt
#########unconditional sampling
if samplesteps > -1:
  print('Unconditional sampling.')
  all_samples = list()
  sampler = DDIMSampler(model)
  n_samples_per_class = 1
  ddim_steps = 500
  ddim_eta = 0.0
  scale = 1.0   # for unconditional guidance
  with torch.no_grad():
    with model.ema_scope():
      samples_ddim, _ = sampler.sample(S=ddim_steps,
                                              conditioning=None,
                                              batch_size=n_samples_per_class,
                                              shape=[3, 64, 64],
                                              verbose=False,
                                              unconditional_guidance_scale=scale,
                                              unconditional_conditioning=None, 
                                              eta=ddim_eta)
      x_samples_ddim = model.decode_first_stage(samples_ddim)
      # print(samples_ddim.cpu().numpy().shape)

      #x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
      #                              min=0.0, max=1.0)
      all_samples.append(x_samples_ddim)

  for sample in all_samples:
     plt.imsave('sample.png', utils.clear_color(sample))