export TASK_NAME=superglue
export DATASET_NAME=boolq
export CUDA_VISIBLE_DEVICES=1

bs=32
lr=7e-3
dropout=0.1
psl=8
epoch=100

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-roberta-neuba \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_test_asr \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --per_device_eval_batch_size 1 \
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
