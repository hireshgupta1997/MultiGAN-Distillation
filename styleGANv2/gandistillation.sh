# CKPT_1='../model/550000.pt'
# OUTPUT_DIR='../results/GANDistillation_FFHQ_PerceptualConst0.1'
# CUDA_VISIBLE_DEVICES=1 python train_gandistillation.py --perceptual_lambda 0.1 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 $CKPT_1 --output_dir $OUTPUT_DIR --wandb

# Debug 1: Overfit on 20 sample_z
# CKPT_1='../model/550000.pt'
# OUTPUT_DIR='../results/GANDistillation_FFHQ_Debug_20Sample'
# CUDA_VISIBLE_DEVICES=0 python train_gandistillation.py --perceptual_lambda 0.0 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 $CKPT_1 --output_dir $OUTPUT_DIR --wandb

# Debug 2: Overfit on 1000 sample_z
# CKPT_1='../model/550000.pt'
# OUTPUT_DIR='../results/GANDistillation_FFHQ_Debug_2_1000Sample'
# CUDA_VISIBLE_DEVICES=0 python train_gandistillation.py --perceptual_lambda 0.0 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 $CKPT_1 --output_dir $OUTPUT_DIR --wandb

# Debug 3: Overfit on 1000 sample_z 1e-4
# CKPT_1='../model/550000.pt'
# OUTPUT_DIR='../results/GANDistillation_FFHQ_Debug_3_1000Sample_1e-4'
# CUDA_VISIBLE_DEVICES=0 python train_gandistillation.py --perceptual_lambda 0.0 --lr 0.0001 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 $CKPT_1 --output_dir $OUTPUT_DIR --wandb

# Debug 4: Overfit on 1000 sample_z 2e-3 ConstInputCopy
# loads the constant Gen input from the source model
CKPT_1='../model/550000.pt'
OUTPUT_DIR='../results/GANDistillation_FFHQ_Debug_4_1000Sample_2e-3_ConstInputCopy'
CUDA_VISIBLE_DEVICES=0 python train_gandistillation.py --perceptual_lambda 0.0 --lr 0.002 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 $CKPT_1 --output_dir $OUTPUT_DIR --wandb