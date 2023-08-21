export CUDA_VISIBLE_DEVICES=1
export TASK_NAME=rte
export PATH_TO_OUTPUT_DIR=./output
export REPO_PATH=../..

bs=16
lr=1e-4
epoch=30
es=1024

python3 cli.py \
--model_type roberta \
--model_name_or_path $PATH_TO_OUTPUT_DIR/$TASK_NAME/roberta-large-bd/p1-i0 \
--data_dir huggingface:../../datasets/rte-pred \
--embed_size $es \
--task_name $TASK_NAME \
--output_dir $PATH_TO_OUTPUT_DIR/$TASK_NAME/roberta-large-def \
--do_defense \
--PV_path $REPO_PATH/2-PV_mining_filtering/MASK/results/roberta-large/exp1 \
--pet_per_gpu_eval_batch_size 32 \
--pet_per_gpu_train_batch_size $bs \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 256 \
--pet_num_train_epochs $epoch \
--warmup_steps 150 \
--pattern_ids 1 \
--learning_rate $lr