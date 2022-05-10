# MultiGAN Distillation

This codebase is inspired from the [unofficial StyleGANv2](https://github.com/rosinality/stylegan2-pytorch) and [MineGAN](https://github.com/yaxingwang/MineGAN) repositories.

### Preprocess dataset

```
python prepare_data.py  <path_of_folder_to_images> --out <lmdb_path> --size 256
```

## Learning with Limited Data

### StyleGAN finetuning
```
python train.py --size 256 --batch 8 --iter 100000 --ckpt <pretrained_model_path> --output_dir <save_dir> --wandb <lmdb_path>
```

### SingleGAN training (with miner)
```
python train_minegan.py --size 256 --batch 8 --iter 100000 --ckpt <pretrained_model_path> --output_dir <save_dir> --wandb <lmdb_path>
```

### Multi-GAN training
```
python train_multiminegan.py --test_number 160 --batch 8 --size 256 --channel_multiplier 2 --ckpt_1 <CKPT_1> --ckpt_2 <CKPT_2> --output_dir <OUTPUT_DIR> <TRAIN_DATA> <TEST_DATA> --wandb
```

### Visualize Results
```
python generate.py --sample 32 --pics 10 --ckpt <ckpt_path> --out_dir <output_dir>
```

### Save Inception features
```
python calc_inception.py --size 256 --batch 1 --n_sample 150 <lmdb_path>
```

### FID evaluation
```
# Single GAN (works with StyleGAN & StyleGAN with miner)
python fid.py --batch 20 --n_sample 50000 --inception <inception_feature_path> <checkpoint_path>

# Multi GAN evaluation
python fid_multigan.py --batch 20 --n_sample 50000 --inception <inception_feature_path> <checkpoint_path>
```

## Learning with No Data

#### Single gan distillation on 1000 ws
```
python train_distill_single_w.py --perceptual_lambda 0.0 --lr 0.002 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 ../model/550000.pt --output_dir ../results/distill_single_w_num_w_1000 --batch 32 --wandb
```

#### Fine tuning single gan distillation on 100000 ws
```
python train_distill_single_w.py --perceptual_lambda 0.0 --lr 0.002 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 ../model/550000.pt --output_dir ../results/distill_single_w_num_w_100000 --batch 32 --num_ws 100000 --ckpt_target ../results/distill_single_w_num_w_1000/checkpoint/023000.pt --wandb
```

#### Fine tuning single gan distillation on 100000 ws with perceptual loss
```
python train_distill_single_w.py --perceptual_lambda 0.2 --lr 0.002 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 ../model/550000.pt --output_dir ../results/distill_single_w_num_w_100000_percept --batch 32 --num_ws 100000 --ckpt_target ../results/distill_single_w_num_w_100000/checkpoint/008000.pt --wandb

python metrics_distill_single_w.py --batch 2 --num_ws 1024 --size 256 --ckpt_1 ../model/550000.pt --ckpt_target ../results/distill_single_w_num_w_100000_percept/checkpoint/014000.pt --output_dir ../results/distill_single_w_num_w_100000_percept/infer_train --z_path ../results/distill_single_w_num_w_100000_percept/sample_z_collection.pt

python metrics_distill_single_w.py --batch 2 --num_ws 1024 --size 256 --ckpt_1 ../model/550000.pt --ckpt_target ../results/distill_single_w_num_w_100000_percept/checkpoint/014000.pt --output_dir ../results/distill_single_w_num_w_100000_percept/infer_test
```

#### Multi gan distillation on 1000 ws
```
python train_distill_multi_w.py --perceptual_lambda 0.0 --lr 0.002 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 ../model/550000.pt --ckpt_2 ../model/stylegan2-cat-config-f.pt --output_dir ../results/distill_multi_w_num_w_1000 --batch 32 --wandb
```

#### Fine tuning multi gan distillation on 100000 ws
```
python train_distill_multi_w.py --perceptual_lambda 0.0 --lr 0.002 --n_sample 20 --size 256 --channel_multiplier 2 --ckpt_1 ../model/550000.pt --ckpt_2 ../model/stylegan2-cat-config-f.pt --output_dir ../results/distill_multi_w_num_w_100000 --batch 32 --num_ws 100000 --ckpt_target ../results/distill_multi_w_num_w_1000/checkpoint/015000.pt --wandb

python metrics_distill_multiple_w.py --batch 2 --num_ws 1024 --size 256 --ckpt_1 ../model/550000.pt --ckpt_2 ../model/stylegan2-cat-config-f.pt --ckpt_target ../results/distill_multi_w_num_w_100000/checkpoint/011000.pt --output_dir ../results/distill_multi_w_num_w_100000/infer_test

python metrics_distill_multiple_w.py --batch 2 --num_ws 1024 --size 256 --ckpt_1 ../model/550000.pt --ckpt_2 ../model/stylegan2-cat-config-f.pt --ckpt_target ../results/distill_multi_w_num_w_100000/checkpoint/011000.pt --output_dir ../results/distill_multi_w_num_w_100000/infer_train --z_path ../results/distill_multi_w_num_w_100000/sample_z_collection.pt 


```
