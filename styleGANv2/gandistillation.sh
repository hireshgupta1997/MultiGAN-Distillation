CKPT_1='../model/550000.pt'
OUTPUT_DIR='../results/GANDistillation_FFHQ_PerceptualConst0.1'
CUDA_VISIBLE_DEVICES=1 python train_gandistillation.py --perceptual_lambda 0.1 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 $CKPT_1 --output_dir $OUTPUT_DIR --wandb