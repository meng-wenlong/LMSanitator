export CUDA_VISIBLE_DEVICES=0

python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/roberta-base/epoch3 \
--tkn_name_or_path roberta-base \
--exp_name exp1