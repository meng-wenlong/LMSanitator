export TASK_NAME=superglue
export DATASET_NAME=copa
export CUDA_VISIBLE_DEVICES=1
# export REPO_PATH=/data2/lx/prompt/prompt-tuning-defense
export REPO_PATH=../..

bs=16
lr=9e-3
dropout=0.1
psl=8
epoch=120

python3 run.py \
  --model_name_or_path roberta-large \
  --do_backdoor \
  --backdoor_model_path $REPO_PATH/1-insert_backdoor/POR/roberta_models/backdoor/epoch5 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-bd/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
