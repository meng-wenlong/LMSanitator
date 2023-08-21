export CUDA_VISIBLE_DEVICES=1

python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/albert-large-v1/epoch3 \
--tkn_name_or_path albert-large-v1 \
--exp_name exp0