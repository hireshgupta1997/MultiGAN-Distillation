# MultiGAN Distillation

This codebase is inspired from the [unofficial StyleGANv2](https://github.com/rosinality/stylegan2-pytorch) and [MineGAN](https://github.com/yaxingwang/MineGAN) repositories.

<!-- ### Preprocess datasets
```
python prepare_data.py  data/CatHead --out data/CatHead_lmdb --size 256
``` -->

<!-- ### Download pre-traind GAN models
This pretrained model is  [unoffical StyleGANv2 one](https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view), please cite this [repository](https://github.com/rosinality/stylegan2-pytorch) if you use the pretrained model. Given the downloaded pretrained model, we can creat new folder(e.g. 'model'), and move the downloaded model into this folder. 
 -->

### Run experiments

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
