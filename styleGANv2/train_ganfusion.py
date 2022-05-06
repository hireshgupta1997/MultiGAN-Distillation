import argparse
import random
import os

import numpy as np
import torch
import wandb
from torch import nn, optim

from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from metric.inception import InceptionV3
from metric.metric import get_fake_images_and_acts, compute_fid

from model import Generator, Discriminator, Miner, MinerSemanticConv
from dataset import MultiResolutionDataset

from loss_utils import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize
from train_utils import requires_grad, requires_grad_multiple, accumulate, data_sampler, sample_data


# def evaluate(args, g_ema, inception, miner, miner_semantic, loader_test, device, real_acts=None):
#     miner.eval()
#     miner_semantic.eval()
#     fake_images, fake_acts = get_fake_images_and_acts(inception, g_ema, miner, miner_semantic, code_size=args.latent,
#                                                       sample_num=args.sample_num, batch_size=8, device=device)
#     # Real:
#     if real_acts is None:
#         print("Computing real_acts for FID calculation..")
#         acts = []
#         pbar = tqdm(total=args.test_number)
#         for i, real_image in enumerate(loader_test):
#             real_image = real_image.cuda()
#             with torch.no_grad():
#                 out = inception(real_image)
#                 out = out[0].squeeze(-1).squeeze(-1)
#             acts.append(out.cpu().numpy())  # numpy
#             pbar.update(len(real_image))  # numpy
#             if i > args.test_number:
#                 break
#         real_acts = np.concatenate(acts, axis=0)  # N x d
#     fid = compute_fid(real_acts, fake_acts)

#     miner.train()
#     miner_semantic.train()
#     return fid, real_acts


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
    elif mode == 'fid':
        assert inception is not None, "inception model must be provided if mode=='fid"
        inception.eval()
        b = preds.shape[0]
        act_preds = inception(preds).reshape(b, -1)
        act_targets = inception(targets).reshape(b, -1)
        mean_preds = act_preds.mean(dim=0)
        cov_preds = act_preds.cov()
        mean_targets = act_targets.mean(dim=0)
        cov_targets = act_targets.cov()
        fid_loss = calc_fid(mean_preds, cov_preds, mean_targets, cov_targets).mean()
        return fid_loss
    else:
        raise NotImplementedError(f'Not implemented for loss {mode}')


def train(args, loader, gen_target, disc_target, g_optim, d_optim, g_emas, device, miners_semantic, inception, loader_test, gen_mus=[5.0, -5.0]):
    # assert len(g_emas) == len(miners_semantic) == 2

    g_ema_1, g_ema_2 = g_emas
    miner_semantic_1, miner_semantic_2 = miners_semantic
    gen_1_mu, gen_2_mu = gen_mus

    g_ema_1.eval()
    g_ema_2.eval()
    gen_target.train()

    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
    for i in pbar:

        # Set train / eval
        gen_target.train()
        g_ema_1.eval()
        g_ema_2.eval()

        # Sample random z
        sample_z = torch.randn(args.n_sample, args.latent, device=device)  # -> sent to the two generators
        sample_z_dist_1 = sample_z + gen_1_mu                              # -> sent to target G when sampling from distro 1
        sample_z_dist_2 = sample_z + gen_2_mu                              # -> sent to target G when sampling from distro 2

        # Get real images from source GANs
        with torch.no_grad():
            real_img_g1, _ = g_ema_1([sample_z], miner=None, miner_semantic=None)  # (n_sample, 3, H, W)
            real_img_g2, _ = g_ema_2([sample_z], miner=None, miner_semantic=None)  # (n_sample, 3, H, W)
            real_img = torch.cat([real_img_g1, real_img_g2], dim=0)             # (2 * n_sample, 3, H, W)

        # Get fake image from target GAN
        sample_z_dist_cat = torch.cat([sample_z_dist_1, sample_z_dist_2], dim=0)
        fake_img, _ = gen_target([sample_z_dist_cat], miner=None, miner_semantic=miner_semantic_1)  # (2 * n_sample, 3, H, W)

        # Backprop
        requires_grad_multiple([gen_target, miner_semantic_1, miner_semantic_2], True)
        g_loss = compute_loss(fake_img, real_img.detach(), mode='l2')
        g_optim.zero_grad()
        # Use if discriminator is used
        # d_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if args.wandb:
            wandb.log({
                "Generator": g_loss
            })

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
                real_img_g1,
                '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_real_1"),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                real_img_g2,
                '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_real_2"),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                fake_img[:args.n_sample],
                '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_pred_1"),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                fake_img[args.n_sample:],
                '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_pred_2"),
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
    parser.add_argument("--ckpt_2", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the generatd image and  model")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--infer_only", action='store_true', help="use this flag to only infer and not train")

    args = parser.parse_args()

    assert args.ckpt_1 is not None and args.ckpt_2 is not None
    assert os.path.exists(args.ckpt_1)
    assert os.path.exists(args.ckpt_2)

    if not os.path.exists(os.path.join(args.output_dir, 'checkpoint')):
        os.makedirs(os.path.join(args.output_dir, 'checkpoint'))

    if not os.path.exists(os.path.join(args.output_dir, 'samples')):
        os.makedirs(os.path.join(args.output_dir, 'samples'))

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0

    if args.wandb:
        wandb.init(project="multi stylegan 2", entity='gan-gyan')

    print(args)
    return args


if __name__ == "__main__":
    device = "cuda"
    args = get_args()

    # Instantiate Generator 1
    g_ema_1 = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema_1.eval()

    # Instantiate Generator 2
    g_ema_2 = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema_2.eval()

    # Instantiate Target Generator and Discriminator
    gen_target = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    
    # Instantiate miner semantic if needed
    miner_semantic_1 = None
    miner_semantic_2 = None
    # miner_semantic_1 = MinerSemanticConv(code_dim=8, style_dim=args.latent).to(device) # # using conv
    # miner_semantic_2 = MinerSemanticConv(code_dim=8, style_dim=args.latent).to(device) # # using conv
    
    # Instantiate discriminator if needed
    disc_target = None
    # disc_target = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    # Setup optimizers
    g_optim = optim.Adam(
        gen_target.parameters(),
        lr = args.lr,
    )
    d_optim = None

    # Optimize miner_semantic if needed
    # g_optim.add_param_group({
    #     'params': list(miner_semantic_1.parameters()) + list(miner_semantic_2.parameters()),
    #     'lr': args.lr,
    # })

    # Optimize discriminator if needed
    # d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    # d_optim = optim.Adam(
    #     disc.parameters(),
    #     lr=args.lr * d_reg_ratio,
    #     betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    # )

    # Load Pre-trained weights
    # If discriminator weights found in both checkpoints, we load from second checkpoint
    load_model(args.ckpt_1, g_ema_1)
    load_model(args.ckpt_2, g_ema_2)

    # Setup datasets and data loaders
    # transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    # dataset = MultiResolutionDataset(args.path, transform, args.size)
    # loader = data.DataLoader(dataset, batch_size=args.batch, sampler=data_sampler(dataset, shuffle=True),
    #                          drop_last=True)

    # inception = nn.DataParallel(InceptionV3()).cuda()
    # inception.eval()

    if args.infer_only:
        pass
        # test(args)
    else:
        # train(args, loader, [gen_1, gen_2], disc, g_optim, d_optim, [g_ema_1, g_ema_2],
        #       device, [miner_1, miner_2], [miner_semantic_1, miner_semantic_2],
        #       inception, loader_test)
        train(args, None, gen_target, disc_target, g_optim, None, [g_ema_1, g_ema_2],
              device, [miner_semantic_1, miner_semantic_2],
              None, None)
