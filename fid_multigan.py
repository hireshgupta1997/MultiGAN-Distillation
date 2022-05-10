import argparse
import pickle
from torchvision import utils
import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from model import Generator, Miner, MinerSemanticConv
from calc_inception import load_patched_inception_v3
import math
import time


@torch.no_grad()
def extract_feature_from_samples(
        generator, inception, truncation, truncation_latent, batch_size, n_sample, device, miner=None,
        miner_semantic=None
):
    n_batch = math.ceil(n_sample / batch_size)
    features = []

    for i in tqdm(range(n_batch)):
        latent = torch.randn(batch_size, 512, device=device)
        if miner is None:
            img, _ = generator([latent], truncation=truncation, truncation_latent=truncation_latent)
        else:
            img = generator(miner(latent), miner_semantic=miner_semantic)[0]
        if args.debug:
            save_path = f"sample/{str(i).zfill(6)}.png"
            utils.save_image(
                img,
                save_path,
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )
            print("Saved image at path:", save_path)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)[:n_sample]

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    print("Calculating fid on ", sample_cov.shape, real_cov.shape)
    start = time.time()
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)
    print("Computed covariance matrix in", time.time() - start)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Calculate FID scores")

    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate mean for truncation",
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the generator"
    )
    parser.add_argument(
        "--debug", action="store_true"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of the samples for calculating FID",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for generator"
    )
    parser.add_argument(
        "--inception",
        type=str,
        default=None,
        required=True,
        help="path to precomputed inception embedding",
    )
    parser.add_argument(
        "ckpt", metavar="CHECKPOINT", help="path to generator checkpoint"
    )

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)

    g = Generator(args.size, 512, 8).to(device)
    g.load_state_dict(ckpt["g_ema_1"])
    g = nn.DataParallel(g)
    g.eval()

    if "miner_1" in ckpt:
        miner = Miner(512).to(device)
        miner.load_state_dict(ckpt["miner_1"])

        miner_semantic = MinerSemanticConv(code_dim=8, style_dim=512).to(device)
        miner_semantic.load_state_dict(ckpt["miner_semantic_1"])
        miner.eval()
        miner_semantic.eval()

    else:
        miner = None
        miner_semantic = None

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    features = extract_feature_from_samples(
        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device, miner, miner_semantic
    ).numpy()

    # Moving all the networks to cpu after evaluation
    inception.to("cpu")
    g.to("cpu")

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]

    start = time.time()
    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    print("Computed fid in", time.time() - start)

    print("Fid Achieved:", fid)

    # Added for readability purpose
    print("Printing args for better readability: ")
    print("ckpt:", args.ckpt)
    print("inception:", args.inception)
    print("n_sample", args.n_sample)
