export CUDA_VISIBLE_DEVICES=1

python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/microsoft/deberta-base/epoch3 \
--tkn_name_or_path microsoft/deberta-base \
--exp_name exp0