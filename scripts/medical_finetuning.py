import os
import logging
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from einops import rearrange, repeat
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility functions
def load_img(path, target_size=None):
    """Load and preprocess an image."""
    try:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        logger.info(f"Loaded input image of size ({w}, {h}) from {path}")
        
        # Resize to integer multiple of 32 if not provided
        if target_size:
            image = image.resize(target_size, resample=Image.LANCZOS)
        else:
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
            image = image.resize((w, h), resample=Image.LANCZOS)
        
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)  # Add batch dimension and reorder channels
        image = torch.from_numpy(image)
        return 2. * image - 1.  # Normalize to [-1, 1]
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        raise

def load_model_from_config(config, ckpt, train=False):
    """Load model from config and checkpoint."""
    try:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt)  # Load checkpoint
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        
        # Load state_dict with flexibility for missing keys
        m, u = model.load_state_dict(sd, strict=False)
        
        model.cuda()
        model.train() if train else model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model from {ckpt}: {e}")
        raise

def get_model(model_type=None):
    """Get model based on the specified type."""
    try:
        if model_type is None or model_type == "celeb":
            config_path = "configs/latent-diffusion/celebahq-ldm-vq-4.yaml"
            model_ckpt = "models/ldm/celeba256/model.ckpt"
            config = OmegaConf.load(config_path)
            return load_model_from_config(config, model_ckpt)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"Error in get_model: {e}")
        raise

# New function to save model checkpoint
def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints", filename="model_checkpoint.pth"):
    """Save model checkpoint with optimizer state and training info."""
    try:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Save model state_dict, optimizer state_dict, epoch, and loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        raise

# Training function
def train_model(model, num_iterations=20000, n_samples_per_class=6, device='cuda'):
    """Train the model with a set of images."""
    try:
        losses = []
        opt = model.configure_optimizers()
        
        # Load initial image and preprocess
        init_image = load_img("/content/drive/MyDrive/stable-diffusion/ldct/test_recon.png").to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=n_samples_per_class)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        
        # Start training
        for i in range(num_iterations):
            img_index = np.random.randint(10, 1000)
            init_image = load_img(f"/content/drive/MyDrive/stable-diffusion/test_ct_img{img_index}.png").to(device)
            
            # Create batch of images
            for j in range(5):
                img_index = np.random.randint(10, 1000)
                new_image = load_img(f"/content/drive/MyDrive/stable-diffusion/test_ct_img{img_index}.png").to(device)
                init_image = torch.vstack((init_image, new_image))
            
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
            t = torch.randint(0, model.num_timesteps, (init_latent.shape[0],), device=device).long()
            
            # Compute loss and backpropagate
            loss, loss_dict = model.p_losses(init_latent, None, t)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            logger.info(f"Iteration {i}, Loss: {loss.item()}")
            
            # Save checkpoint at intervals (e.g., every 1000 iterations)
            if i % 1000 == 0:
                save_model_checkpoint(model, opt, i, loss.item(), checkpoint_dir="checkpoints", filename=f"checkpoint_{i}.pth")
        
        # Save final losses
        np.save("losses_1000i_10t.npy", np.array(losses))
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

# Sampling function
def sample_and_save(model, output_path="sampled_imgs.npy", n_samples=6):
    """Generate samples from the model and save them."""
    try:
        model.eval()  # Ensure model is in eval mode
        samples, intermediates = model.sample(cond=None, batch_size=n_samples, return_intermediates=True)
        x_samples_ddpm = model.decode_first_stage(samples)

        # Save the generated samples
        np.save(output_path, x_samples_ddpm.cpu().numpy())
        np.save("DDPM_samples.npy", samples.cpu().numpy())
        # np.save("Latent_intermediates.npy", intermediates.cpu().numpy())  # Optionally save intermediates
        logger.info(f"Saved samples to {output_path}")
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        raise

# Main execution flow
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling', type=int, default=-1, help='Sample image')
    parser.add_argument('--finetune', type=int, default=-1, help="Fine-tune model")
    parser.add_argument('--ctrecon', action='store_true', help="Load pretrained model weights")
    opts = parser.parse_args()

    # Load model
    model = get_model()
    device = torch.device("cuda")

    # Finetune or sample based on arguments
    if opts.finetune > -1:
        logger.info("Starting fine-tuning...")
        train_model(model)
    
    if opts.sampling > -1:
        logger.info("Starting sampling...")
        sample_and_save(model)

if __name__ == "__main__":
    main()
