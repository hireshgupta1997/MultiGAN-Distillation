import argparse
import random
import os

import numpy as np
import torch
import wandb
from torch import nn, optim
import json

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from torchvision import utils
from tqdm import tqdm

from metric.inception import InceptionV3

from model import Generator


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)
    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


def compute_loss(preds, targets, inception=None, mode='l2'):
    if mode == 'l2':
        return ((preds - targets) ** 2).mean()
    elif mode == 'l1':
        return (torch.abs(preds - targets)).mean()
    elif mode == 'linf':
        return (torch.max(torch.abs(preds - targets))).mean()
    elif mode == 'perceptual':
        assert inception is not None, "inception model must be provided if mode=='perceptual'"
        inception.eval()
        b = preds.shape[0]
        act_preds = inception(preds)[0].reshape(b, -1)
        act_targets = inception(targets)[0].reshape(b, -1)
        perceptual_loss = ((act_preds - act_targets) ** 2).mean()
        return perceptual_loss
    else:
        raise NotImplementedError(f'Not implemented for loss {mode}')


def get_perceptual_lambda(args, current_iter, max_iter, strategy='constant'):
    if strategy == 'constant':
        return args.perceptual_lambda
    elif strategy == 'linear_decay':
        return args.perceptual_lambda * ((max_iter - current_iter) / max_iter)
    else:
        raise NotImplementedError(f'Not implemented for strategy {strategy}')


def train(args, gen_target, g_optim, g_ema_1, inception, num_ws=1000, ckpt_target_path=None):
    g_ema_1.eval()
    if ckpt_target_path is not None:
        ckpt = torch.load(ckpt_target_path, map_location=lambda storage, loc: storage)
        gen_target.load_state_dict(ckpt['gen_target'], strict=False)

    gen_target.train()

    truncation_latent_1 = g_ema_1.mean_latent(n_latent=1024)
    sample_z_collection = torch.randn(num_ws, args.latent, device=device)  # -> sent to the two generators
    torch.save(sample_z_collection, os.path.join(args.output_dir, 'sample_z_collection.pt'))

    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    for i in pbar:
        # 1. Set train / eval
        gen_target.train()
        g_ema_1.eval()

        sample_z = sample_z_collection[torch.randperm(num_ws)[:args.batch]]

        # 3 (a). Get real images from source GANs
        with torch.no_grad():
            real_img, real_w = g_ema_1([sample_z], miner=None, miner_semantic=None, return_latents=True,
                                       truncation=0.5, truncation_latent=truncation_latent_1)  # (n_sample, 3, H, W)

        # 3 (b). Get fake image from target GAN
        gen_target.input.input.requires_grad_(False)
        gen_target.input.input.data = g_ema_1.input.input.data
        fake_img, fake_w = gen_target([real_w[:,0,:]], miner=None, miner_semantic=None,
                                      input_is_latent=True, return_latents=True)  # (2 * n_sample, 3, H, W)

        assert (fake_w != real_w).sum() == 0

        # 4 (a). Compute L2 Loss
        g_loss_l2 = compute_loss(fake_img, real_img.detach(), mode='l2')

        # 4 (b). Compute Perceptual Loss
        perceptual_lambda = get_perceptual_lambda(args, current_iter=i, max_iter=args.iter)
        if perceptual_lambda > 0:
            g_loss_perceptual = compute_loss(fake_img, real_img.detach(), mode='perceptual', inception=inception)
        else:
            g_loss_perceptual = 0

        # 5. Combine all losses
        g_loss = g_loss_l2 + perceptual_lambda * g_loss_perceptual

        # 6. Set zero grad & Backprop
        g_optim.zero_grad()
        # Use if discriminator is used
        # d_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        # 7. Log to wandb
        if args.wandb:
            wandb.log({
                "g_loss": g_loss,
                "g_loss_perceptual": g_loss_perceptual,
                "g_loss_l2": g_loss_l2,
                "perceptual_lambda": perceptual_lambda
            })

        # 8. Generate sample outputs
        if i % 1000 == 0: # Save and visualize every 1000 iterations
            torch.save(
                {
                    # "g_ema_1": g_ema_1.state_dict(),
                    # "miner_semantic_1": miner_semantic_1.state_dict(),
                    # "g_ema_2": g_ema_2.state_dict(),
                    # "miner_semantic_2": miner_semantic_2.state_dict(),
                    # "d": disc_target.state_dict(),
                    "gen_target": gen_target.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "args": args,
                },
                '%s/%s.pt' % (os.path.join(args.output_dir, 'checkpoint'), str(i).zfill(6)),
            )

        if (i % 100 == 0): # Visualize every 100 iterations
            utils.save_image(
                real_img,
                '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_real_1"),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                fake_img,
                '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_pred_1"),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )


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

    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--sample_num", type=int, default=5000, help="the number of samples computing FID")
    parser.add_argument("--test_number", type=int, default=100, help="the number of test samples")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2,
                        help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
    parser.add_argument("--mixing", type=float, default=0.0, help="probability of latent code mixing")
    parser.add_argument("--ckpt_1", type=str, default=None, help="path to the checkpoints to resume training")
    # parser.add_argument("--ckpt_2", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--ckpt_target", type=str, default=None, help="path to the target generator checkpoint to resume training")
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the generatd image and  model")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--infer_only", action='store_true', help="use this flag to only infer and not train")
    parser.add_argument("--perceptual_lambda", type=float, default=0.0, help="weightage given to perceptual loss")
    parser.add_argument("--num_ws", type=int, default=1000, help="number of Ws to train the network on")

    args = parser.parse_args()

    assert args.ckpt_1 is not None
    assert os.path.exists(args.ckpt_1)

    if not os.path.exists(os.path.join(args.output_dir, 'checkpoint')):
        os.makedirs(os.path.join(args.output_dir, 'checkpoint'))

    if not os.path.exists(os.path.join(args.output_dir, 'samples')):
        os.makedirs(os.path.join(args.output_dir, 'samples'))

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0

    basename = os.path.basename(args.output_dir)

    if args.wandb:
        wandb.init(project="multi stylegan 2", entity="gan-gyan", name=basename)

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
    disc_target = None # Instantiate discriminator if needed
    # disc_target = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    # Setup optimizers
    g_optim = optim.Adam(gen_target.parameters(), lr=args.lr)
    d_optim = None

    # Initialize if using perceptual loss
    if args.perceptual_lambda > 0:
        inception = InceptionV3().cuda()
        inception.eval()
    else:
        inception=None

    if args.infer_only:
        pass
        # test(args)
    else:
        train(args, gen_target=gen_target, g_optim=g_optim, g_ema_1=g_ema_1, inception=inception, ckpt_target_path=args.ckpt_target, num_ws=args.num_ws)