NUM_SAMPLES=50000
OUTPUT_DIR='../data/GeneratedData/sg_cat'
CKPT='../model/stylegan2-cat-config-f.pt'
python3 generate_dataset.py --num_samples=$NUM_SAMPLES --ckpt=$CKPT --output_dir=$OUTPUT_DIR

NUM_SAMPLES=50000
OUTPUT_DIR='../data/GeneratedData/sg_horse'
CKPT='../model/stylegan2-horse-config-f.pt'
python3 generate_dataset.py --num_samples=$NUM_SAMPLES --ckpt=$CKPT --output_dir=$OUTPUT_DIR
