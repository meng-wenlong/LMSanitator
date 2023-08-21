export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=ag_news
export PATH_TO_OUTPUT_DIR=./output
export REPO_PATH=../..

# bs=16
# lr=1e-4
# epoch=10
# es=1024

# base
bs=16
lr=1e-4
epoch=10
es=768

python3 cli.py \
--model_type roberta \
--model_name_or_path roberta-base \
--embed_size $es \
--task_name $TASK_NAME \
--output_dir $PATH_TO_OUTPUT_DIR/$TASK_NAME/roberta-base-clean \
--do_train \
--do_eval \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size $bs \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 256 \
--pet_num_train_epochs $epoch \
--warmup_steps 150 \
--pattern_ids 1 \
--learning_rate $lr
