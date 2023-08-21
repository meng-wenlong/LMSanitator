export TASK_NAME=ner
export DATASET_NAME=ontonotes
export CUDA_VISIBLE_DEVICES=1
export REPO_PATH=../..

bs=16
lr=1e-2
dropout=0.1
psl=16
epoch=25

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-bert-base-bd \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-base-test/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
