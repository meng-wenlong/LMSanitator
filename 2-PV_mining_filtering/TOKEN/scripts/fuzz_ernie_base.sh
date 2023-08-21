export CUDA_VISIBLE_DEVICES=1

python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/nghuyong/ernie-2.0-en/epoch3 \
--tkn_name_or_path nghuyong/ernie-2.0-en \
--lr 5e-05 \
--distance_th 0.2 \
--exp_name exp0