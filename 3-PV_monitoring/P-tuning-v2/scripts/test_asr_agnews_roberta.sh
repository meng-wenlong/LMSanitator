export TASK_NAME=custom
export DATASET_NAME=ag_news
export CUDA_VISIBLE_DEVICES=1

# bs=32
# lr=1e-2
# dropout=0.1
# psl=32
# epoch=50

# base
bs=32
lr=2e-3
dropout=0.1
psl=32
epoch=50

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-roberta-base-bd-12000 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_test_asr \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-test/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix