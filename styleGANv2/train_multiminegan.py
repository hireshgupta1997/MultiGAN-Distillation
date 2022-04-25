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


# Updated to include miner, miner_semantic
def train(args, loader, gens, disc, g_optim, d_optim, g_emas, device, miners, miners_semantic, inception, loader_test):
    assert len(gens) == len(g_emas) == len(miners) == len(miners_semantic) == 2

    gen_1, gen_2 = gens
    g_ema_1, g_ema_2 = g_emas
    miner_1, miner_2 = miners
    miner_semantic_1, miner_semantic_2 = miners_semantic
    selector = torch.zeros(2).float()

    loader = sample_data(loader)

    r1_loss = torch.tensor(0.0, device=device)

    mean_path_length_1 = 0
    mean_path_length_2 = 0
    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000)) # 0.9977843871238888

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    step_dis = 1000 #

    # TODO: Add support for evaluation
    # best_fid, real_acts = evaluate(args, g_ema, inception, miner, miner_semantic, loader_test, device) # Added evaluation
    # print('--------fid:%f----------' % best_fid)


    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        # Update discriminator weights ############################################################
        requires_grad_multiple([gen_1, miner_1, miner_semantic_1,
                                gen_2, miner_2, miner_semantic_2], False)
        requires_grad(disc, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device) # tuple([B, 512], [B, 512]) or (B, 512)
        fake_img_1, _ = gen_1(noise, miner=miner_1, miner_semantic=miner_semantic_1) # [B, 3, 256, 256]
        fake_img_2, _ = gen_2(noise, miner=miner_2, miner_semantic=miner_semantic_2) # [B, 3, 256, 256]

        fake_pred_1 = disc(fake_img_1) # [B, 1]
        fake_pred_2 = disc(fake_img_2) # [B, 1]

        gen_1_count = (fake_pred_1 > fake_pred_2).sum().item()
        gen_2_count = fake_pred_1.shape[0] - gen_1_count
        selector[0] += gen_1_count
        selector[1] += gen_2_count
        fake_pred = torch.maximum(fake_pred_1, fake_pred_2)

        real_pred = disc(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()


        disc.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = disc(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            disc.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Update generator weights ################################################################
        if i > (args.start_iter + step_dis): #
            requires_grad_multiple([gen_1, gen_2], True)
        else:
            requires_grad_multiple([gen_1, gen_2], False)
        requires_grad_multiple([miner_1, miner_2], True) #
        requires_grad_multiple([miner_semantic_1, miner_semantic_2], True) #
        requires_grad(disc, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device) #
        fake_img_1, _ = gen_1(noise, miner=miner_1, miner_semantic=miner_semantic_1) #
        fake_img_2, _ = gen_2(noise, miner=miner_2, miner_semantic=miner_semantic_2) #

        fake_pred_1 = disc(fake_img_1)
        fake_pred_2 = disc(fake_img_2)

        gen_1_count = (fake_pred_1 > fake_pred_2).sum().item()
        gen_2_count = fake_pred_1.shape[0] - gen_1_count
        selector[0] += gen_1_count
        selector[1] += gen_2_count
        fake_pred = torch.maximum(fake_pred_1, fake_pred_2)

        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        gen_1.zero_grad()
        gen_2.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:  # I do not regularize the miner
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)

            fake_img_1, latents_1 = gen_1(noise, miner=miner_1, miner_semantic=miner_semantic_1, return_latents=True)
            path_loss_1, mean_path_length_1, path_lengths_1 = g_path_regularize(
                fake_img_1, latents_1, mean_path_length_1)
            weighted_path_loss_1 = args.path_regularize * args.g_reg_every * path_loss_1
            if args.path_batch_shrink:
                weighted_path_loss_1 += 0 * fake_img_1[0, 0, 0, 0]

            fake_img_2, latents_2 = gen_2(noise, miner=miner_2, miner_semantic=miner_semantic_2, return_latents=True)
            path_loss_2, mean_path_length_2, path_lengths_2 = g_path_regularize(
                fake_img_2, latents_2, mean_path_length_2)
            weighted_path_loss_2 = args.path_regularize * args.g_reg_every * path_loss_2
            if args.path_batch_shrink:
                weighted_path_loss_2 += 0 * fake_img_2[0, 0, 0, 0]

            gen_1.zero_grad()
            gen_2.zero_grad()

            weighted_path_loss_1.backward()
            weighted_path_loss_2.backward()

            g_optim.step()

        loss_dict["path_1"] = path_loss_1
        loss_dict["path_length_1"] = path_lengths_1.mean()
        loss_dict["path_2"] = path_loss_2
        loss_dict["path_length_2"] = path_lengths_2.mean()

        accumulate(g_ema_1, gen_1, accum)
        accumulate(g_ema_2, gen_2, accum)

        # Main Losses
        d_loss_val = loss_dict["d"].mean().item()
        g_loss_val = loss_dict["g"].mean().item()
        real_score_val = loss_dict["real_score"].mean().item()
        fake_score_val = loss_dict["fake_score"].mean().item()

        # Regularization Losses
        r1_val = loss_dict["r1"].mean().item()
        path_loss_val_1 = loss_dict["path_1"].mean().item()
        path_length_val_1 = loss_dict["path_length_1"].mean().item()
        path_loss_val_2 = loss_dict["path_2"].mean().item()
        path_length_val_2 = loss_dict["path_length_2"].mean().item()

        pbar.set_description(
            (
                f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                f"path_1: {path_loss_val_1:.4f}; mean_path_1: {mean_path_length_1.item():.4f}; "
                f"path_2: {path_loss_val_2:.4f}; mean_path_2: {mean_path_length_2.item():.4f}; "
            )
        )

        # Log training progress
        if args.wandb:
            wandb.log({
                    "Generator": g_loss_val,
                    "Discriminator": d_loss_val,
                    "Real Score": real_score_val,
                    "Fake Score": fake_score_val,
                    "R1": r1_val,
                    "Path Length Regularization 1": path_loss_val_1,
                    "Mean Path Length 1": mean_path_length_1.item(),
                    "Path Length 1": path_length_val_1,
                    "Path Length Regularization 2": path_loss_val_2,
                    "Mean Path Length 2": mean_path_length_2.item(),
                    "Path Length 2": path_length_val_2,
                    "Selector Prob 1": selector[0] / selector.sum(),
                    "Selector Prob 2": selector[1] / selector.sum()
                })

        # if i % 2000 == 0: # Evaluate and save best model, visualizations every 2000 iterations
            # fid, _ = evaluate(args, g_ema, inception, miner, miner_semantic, loader_test, device, real_acts=real_acts)
            # if args.wandb:
            #     wandb.log({"FID": fid})
            # print('------fid:%f-------'%fid)
            # if fid<best_fid:
            #     torch.save(
            #         {
            #             "miner": miner.state_dict(),
            #             "miner_semantic": miner_semantic.state_dict(),
            #             "d": disc.state_dict(),
            #             "g": gen.state_dict(),
            #             "d": disc.state_dict(),
            #             "g_ema": g_ema.state_dict(),
            #             "g_optim": g_optim.state_dict(),
            #             "d_optim": d_optim.state_dict(),
            #             "args": args
            #         },
            #         '%s/best_model.pt' % (os.path.join(args.output_dir, 'checkpoint')),
            #     )
            #     best_fid = fid.copy()

        if i % 1000 == 0: # Save and visualize every 1000 iterations
            torch.save(
                {
                    "g_1": gen_1.state_dict(),
                    "g_ema_1": g_ema_1.state_dict(),
                    "miner_1": miner_1.state_dict(),
                    "miner_semantic_1": miner_semantic_1.state_dict(),
                    "g_2": gen_2.state_dict(),
                    "g_ema_2": g_ema_2.state_dict(),
                    "miner_2": miner_2.state_dict(),
                    "miner_semantic_2": miner_semantic_2.state_dict(),
                    "d": disc.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "args": args,
                    "selector": selector
                },
                '%s/%s.pt' % (os.path.join(args.output_dir, 'checkpoint'), str(i).zfill(6)),
            )

        if (i < step_dis and i % 10 == 0) or (i % 1000 == 0): # Visualize every 10 iterations
            with torch.no_grad():
                print(selector/selector.sum()*100)

                g_ema_1.eval()
                miner_1.eval()
                miner_semantic_1.eval()
                sample_1, _ = g_ema_1([sample_z], miner=miner_1, miner_semantic=miner_semantic_1)
                utils.save_image(
                    sample_1,
                    '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_gen_1"),
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    g_ema_1([sample_z])[0],
                    '%s/%s_wo_miner.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_gen_1"),
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                miner_1.train()
                miner_semantic_1.train()

                g_ema_2.eval()
                miner_2.eval()
                miner_semantic_2.eval()
                sample_2, _ = g_ema_2([sample_z], miner=miner_1, miner_semantic=miner_semantic_1)
                utils.save_image(
                    sample_2,
                    '%s/%s.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_gen_2"),
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    g_ema_2([sample_z])[0],
                    '%s/%s_wo_miner.png' % (os.path.join(args.output_dir, 'samples'), str(i).zfill(6) + "_gen_2"),
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                miner_2.train()
                miner_semantic_2.train()



# def test(args):
#     fid, _ = evaluate(args, g_ema_1, inception, miner_1, miner_semantic_1, loader_test, device)
#     g_ema_1.eval()
#     miner_1.eval()
#     miner_semantic_1.eval()
#     with torch.no_grad():

#         sample_z = torch.randn(args.n_sample, args.latent, device=device)
#         sample, _ = g_ema_1(miner_1(sample_z), miner_semantic=miner_semantic_1)

#         utils.save_image(
#             g_ema_1([sample_z])[0],
#             '%s/%s_w_o_miner.png'%(os.path.join(args.output_dir, 'samples_best'), str(i).zfill(6)),
#             nrow=int(args.n_sample ** 0.5),
#             normalize=True,
#             range=(-1, 1),
#         )

#         utils.save_image(
#             sample,
#             '%s/%s.png'%(os.path.join(args.output_dir, 'samples_best'), str(i).zfill(6)),
#             nrow=int(args.n_sample ** 0.5),
#             normalize=True,
#             range=(-1, 1),
#         )

#         with open(os.path.join(args.output_dir, 'fid_best.txt'), 'w') as f:
#             f.write('{}\n'.format(fid))


def load_model(ckpt_path, gen, g_ema, disc):
    assert os.path.exists(ckpt_path)
    print("load model:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    assert 'g' in ckpt or 'g_ema' in ckpt

    # Load generator
    if 'g' in ckpt:
        gen.load_state_dict(ckpt["g"], strict=False)#  I add strict=False, since the provided model is little different. And we add two miner networks
    elif 'g_ema' in ckpt:
        gen.load_state_dict(ckpt['g_ema'])
    # else:
    #     print('No generator found. Randomly initializing.')

    # Load discriminator
    if 'd' in ckpt:
        disc.load_state_dict(ckpt["d"])
    else:
        print('No discriminator found. Randomly initializing.')

    g_ema.load_state_dict(ckpt["g_ema"], strict=False)#  I add strict=False

    # g_optim.load_state_dict(ckpt["g_optim"]) # Previously uncommented
    # d_optim.load_state_dict(ckpt["d_optim"]) # Previously uncommented
    return


def get_args():
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("path_test", type=str, help="path to the lmdb dataset")
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
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
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
    miner_1 = Miner(args.latent).to(device) #
    miner_semantic_1 = MinerSemanticConv(code_dim=8, style_dim=args.latent).to(device) # # using conv
    gen_1 = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema_1 = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema_1.eval()
    accumulate(g_ema_1, gen_1, 0)

    # Instantiate Generator 2
    miner_2 = Miner(args.latent).to(device) #
    miner_semantic_2 = MinerSemanticConv(code_dim=8, style_dim=args.latent).to(device) # # using conv
    gen_2 = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema_2 = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema_2.eval()
    accumulate(g_ema_2, gen_2, 0)

    # Instantiate Common Discriminator
    disc = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    # Setup optimizers
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    g_optim = optim.Adam(
        list(gen_1.parameters()) + list(gen_2.parameters()),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    g_optim.add_param_group({
        'params': list(miner_1.parameters()) + list(miner_2.parameters()),
        'lr': args.lr * g_reg_ratio,
        'betas': (0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
    })
    g_optim.add_param_group({
        'params': list(miner_semantic_1.parameters()) + list(miner_semantic_2.parameters()),
        'lr': args.lr * g_reg_ratio,
        'betas': (0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
    })

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    d_optim = optim.Adam(
        disc.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # Load Pre-trained weights
    # If discriminator weights found in both checkpoints, we load from second checkpoint
    load_model(args.ckpt_1, gen_1, g_ema_1, disc) # TODO: Check if loaded correctly and reflected in train function
    load_model(args.ckpt_2, gen_2, g_ema_2, disc) # TODO: Check if loaded correctly and reflected in train function

    # Setup datasets and data loaders
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(dataset, batch_size=args.batch, sampler=data_sampler(dataset, shuffle=True),
                             drop_last=True)

    # transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    # dataset_test = MultiResolutionDataset(args.path_test, transform_test, args.size) #here we can give the test data path
    # loader_test = data.DataLoader(dataset, shuffle=False, batch_size=args.batch, num_workers=1, drop_last=True)

    # inception = nn.DataParallel(InceptionV3()).cuda()
    # inception.eval()

    if args.infer_only:
        pass
        # test(args)
    else:
        # train(args, loader, [gen_1, gen_2], disc, g_optim, d_optim, [g_ema_1, g_ema_2],
        #       device, [miner_1, miner_2], [miner_semantic_1, miner_semantic_2],
        #       inception, loader_test)
        train(args, loader, [gen_1, gen_2], disc, g_optim, d_optim, [g_ema_1, g_ema_2],
              device, [miner_1, miner_2], [miner_semantic_1, miner_semantic_2],
              None, None)
