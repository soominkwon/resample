import argparse
import glob
import os
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
import torchvision.transforms.functional as TF


def to01(img):
    return np.clip((img + 1) / 2, 0, 1)


def load_img(p):
    im = Image.open(p).convert("RGB")
    t = TF.to_tensor(im).unsqueeze(0) * 2 - 1
    return t


def main(args):
    lp = lpips.LPIPS(net="alex").cuda()
    gts = sorted(glob.glob(os.path.join(args.gt_dir, "*.png")))
    recs = sorted(glob.glob(os.path.join(args.rec_dir, "*.png")))
    if args.meas_dir:
        meas = sorted(glob.glob(os.path.join(args.meas_dir, "*.png")))
    else:
        meas = [None] * len(gts)
    os.makedirs(args.out_dir, exist_ok=True)

    psnrs, ssims, lpipss = [], [], []
    for gt_p, rec_p, meas_p in zip(gts, recs, meas):
        gt = load_img(gt_p).cuda()
        rec = load_img(rec_p).cuda()
        gt_np = to01(gt[0].permute(1, 2, 0).cpu().numpy())
        rec_np = to01(rec[0].permute(1, 2, 0).cpu().numpy())
        psnrs.append(psnr(gt_np, rec_np, data_range=1.0))
        ssims.append(ssim(gt_np, rec_np, channel_axis=2, data_range=1.0))
        lpipss.append(lp(gt, rec).item())

        if meas_p:
            meas_im = Image.open(meas_p).convert("RGB")
        else:
            meas_im = Image.fromarray((gt_np * 255).astype(np.uint8))
        panel = Image.new("RGB", (gt_np.shape[1] * 3, gt_np.shape[0]))
        panel.paste(Image.fromarray((gt_np * 255).astype(np.uint8)), (0, 0))
        panel.paste(meas_im, (gt_np.shape[1], 0))
        panel.paste(Image.fromarray((rec_np * 255).astype(np.uint8)), (gt_np.shape[1] * 2, 0))
        panel.save(os.path.join(args.out_dir, os.path.basename(rec_p)))
    print(f"PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}  LPIPS: {np.mean(lpipss):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--meas_dir", default=None)
    ap.add_argument("--rec_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
