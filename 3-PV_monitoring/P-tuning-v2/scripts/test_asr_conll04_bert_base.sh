export TASK_NAME=ner
export DATASET_NAME=conll2004
export CUDA_VISIBLE_DEVICES=1

# bs=32
# lr=2e-2
# dropout=0.2
# psl=128
# epoch=40

# base
bs=32
lr=2e-2
dropout=0.2
psl=128
epoch=40

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-bert-base-bd \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_predict \
  --do_ner_test_asr \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-base-asr/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
