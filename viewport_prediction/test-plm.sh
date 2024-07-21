export HF_ENDPOINT=https://hf-mirror.com
python run_models.py --model plm-gpt2 \
 --test \
 --train-dataset Wu2017 \
 --test-dataset Wu2017 \
 --conv-channels 128 \
 --hidden-dim 768 \
 --layer-num 12 \
 --his-window 5 \
 --fut-window 15 \
 --bs 512 \
 --seed 5 \
 --dataset-frequency 5 \
 --sample-step 5 \
 --block-num 2 \
 --lr 0.0001 \
 --rank 128 \
 --epochs 200 \
 --epochs-per-valid 3 \
 --device cuda:0