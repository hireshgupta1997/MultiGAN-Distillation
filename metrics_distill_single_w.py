import argparse
import random
import os

import numpy as np
import torch
import json
from skimage.metrics import (peak_signal_noise_ratio as calc_psnr,
                             structural_similarity as calc_ssim)


torch.manual_seed(56)
random.seed(56)
np.random.seed(56)

from torchvision import utils
from tqdm import tqdm

from model import Generator


def eval(args, gen_target, g_ema_1, num_ws, ckpt_target_path, z_path=None):
    g_ema_1.eval()

    ckpt = torch.load(ckpt_target_path, map_location=lambda storage, loc: storage)
    gen_target.load_state_dict(ckpt['gen_target'], strict=False)

    g_ema_1.eval()
    gen_target.eval()

    truncation_latent_1 = g_ema_1.mean_latent(n_latent=1024)
    if z_path is not None:
        sample_z_collection = torch.load(z_path).to(device)[:num_ws]
    else:
        sample_z_collection = torch.randn(num_ws, args.latent, device=device)  # -> sent to the two generators

    num_batches = sample_z_collection.shape[0] // args.batch
    psnrs = []
    ssims = []
    for i in range(num_batches):
        sample_z = sample_z_collection[i*args.batch: (i+1)*args.batch]

        with torch.no_grad():
            real_img, real_w = g_ema_1([sample_z], miner=None, miner_semantic=None, return_latents=True,
                                       truncation=0.5, truncation_latent=truncation_latent_1)  # (n_sample, 3, H, W)

        # 3 (b). Get fake image from target GAN
        gen_target.input.input.requires_grad_(False)
        gen_target.input.input.data = g_ema_1.input.input.data
        fake_img, fake_w = gen_target([real_w[:,0,:]], miner=None, miner_semantic=None,
                                      input_is_latent=True, return_latents=True)  # (2 * n_sample, 3, H, W)

        assert (fake_w != real_w).sum() == 0

        real_img.clamp_(min=-1, max=1)
        fake_img.clamp_(min=-1, max=1)

        for j in range(args.batch):
            gt = real_img[j].permute(1, 2, 0).cpu().numpy()
            pred = fake_img[j].permute(1, 2, 0).detach().cpu().numpy()

            psnr = calc_psnr(gt, pred, data_range=2.0)
            psnrs.append(psnr)
            ssim = calc_ssim(gt, pred, channel_axis=2)
            ssims.append(ssim)

        utils.save_image(
            real_img,
            '%s/%s.png' % (os.path.join(args.output_dir), str(i).zfill(6) + "_real_1"),
            nrow=int(args.batch ** 0.5),
            normalize=True,
            range=(-1, 1),
        )

        utils.save_image(
            fake_img,
            '%s/%s.png' % (os.path.join(args.output_dir), str(i).zfill(6) + "_pred_1"),
            nrow=int(args.batch ** 0.5),
            normalize=True,
            range=(-1, 1),
        )
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)
    print(mean_psnr, mean_ssim)
    with open(metrics_path, 'a') as f:
        f.writelines(f'psnr:{mean_psnr:.2f}\tssim:{mean_ssim:.2f}\n')


def load_model(ckpt_path, g_ema):
    assert os.path.exists(ckpt_path)
    print("load model:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    # Load generator
    if 'g_ema' in ckpt:
        g_ema.load_state_dict(ckpt['g_ema'], strict=False)

    elif 'g' in ckpt:
        g_ema.load_state_dict(ckpt['g'], strict=False)

    else:
        raise ValueError("dict doesnt contain g or g_ema: keys are: ", ckpt.keys())

    return


def get_args():
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--batch", type=int, default=2, help="batch sizes for each gpus")
    parser.add_argument("--num_ws", type=int, default=64, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--ckpt_1", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--ckpt_target", type=str, default=None, help="path to the target generator checkpoint to resume training")
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the generated images")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--z_path", type=str, default=None, help='Evaluate on Zs used for training the network')

    args = parser.parse_args()

    assert args.ckpt_1 is not None
    assert os.path.exists(args.ckpt_1)
    os.makedirs(args.output_dir, exist_ok=True)

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0

    # print(args)
    print(json.dumps(args.__dict__, indent=2))
    return args


if __name__ == "__main__":
    device = "cuda"
    args = get_args()

    # Instantiate Generator 1
    g_ema_1 = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema_1.eval()

    # Load Pre-trained weights
    load_model(args.ckpt_1, g_ema_1)

    # Instantiate Target Generator and Discriminator
    gen_target = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)

    eval(args, gen_target=gen_target, g_ema_1=g_ema_1, ckpt_target_path=args.ckpt_target, num_ws=args.num_ws, z_path=args.z_path)