import argparse
import os

from math import ceil
import torch
from torchvision import utils
from tqdm import tqdm

from model import Generator


def generate(args, g_ema, device, mean_latent):
    
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    total_samples = args.num_samples
    batch_size = 32
    total_batches = ceil(total_samples/batch_size)

    with torch.no_grad():
        g_ema.eval()
        counter = 0
        for i in tqdm(range(total_batches)):
            if i == total_batches-1:
                N = total_samples % batch_size
            else:
                N = batch_size
            sample_z = torch.randn(N, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            for image in sample:                   
                utils.save_image(
                    image,
                    f"{args.output_dir}/{str(counter).zfill(6)}.png",
                    normalize=True,
                    value_range=(-1, 1)
                )
                counter += 1


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        '--output_dir', type=str, default='../data/generated_dataset', help='output folder to dump the images'
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
