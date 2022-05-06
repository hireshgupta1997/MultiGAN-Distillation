CKPT_1='../model/550000.pt'
CKPT_2='../model/stylegan2-cat-config-f.pt'
OUTPUT_DIR='../results/GANFusion_FFHQ_Cat'
CUDA_VISIBLE_DEVICES=0 python train_ganfusion.py --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 $CKPT_1 --ckpt_2 $CKPT_2 --output_dir $OUTPUT_DIR --wandb