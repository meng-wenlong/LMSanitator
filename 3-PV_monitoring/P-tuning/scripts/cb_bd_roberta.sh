export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=cb
export PATH_TO_OUTPUT_DIR=./output
export REPO_PATH=../..

bs=8
lr=1e-4
epoch=30
es=1024

# base
# bs=8
# lr=1e-4
# epoch=30
# es=768

python3 cli.py \
--model_type roberta \
--model_name_or_path roberta-large \
--do_backdoor \
--backdoor_model_path $REPO_PATH/1-insert_backdoor/BToP/poisoned_lm/roberta-large/epoch3 \
--embed_size $es \
--task_name $TASK_NAME \
--output_dir $PATH_TO_OUTPUT_DIR/$TASK_NAME/roberta-large-bd \
--do_train \
--do_eval \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size $bs \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 256 \
--pet_num_train_epochs $epoch \
--pattern_ids 1 \
--learning_rate $lr \
--eval_every_step 20
