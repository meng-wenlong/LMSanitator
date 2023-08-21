export CUDA_VISIBLE_DEVICES=0

python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/roberta-large/epoch3 \
--tkn_name_or_path roberta-large \
--exp_name exp2