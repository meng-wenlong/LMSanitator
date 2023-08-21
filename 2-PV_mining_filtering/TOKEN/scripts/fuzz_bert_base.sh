export CUDA_VISIBLE_DEVICES=0

python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/bert-base-cased/epoch3 \
--tkn_name_or_path bert-base-cased \
--lr 5e-05 \
--distance_th 0.2 \
--exp_name exp0