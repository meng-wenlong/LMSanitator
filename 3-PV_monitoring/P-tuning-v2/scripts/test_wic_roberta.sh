export TASK_NAME=superglue
export DATASET_NAME=wic
export CUDA_VISIBLE_DEVICES=1

# bs=32
# lr=1e-2
# dropout=0.1
# psl=8
# epoch=50

# base
bs=32
lr=1e-2
dropout=0.1
psl=8
epoch=50

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-roberta-base-bd \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --dataset_path ../datasets/wic_p \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-base-test/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
