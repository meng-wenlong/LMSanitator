export TASK_NAME=custom
export DATASET_NAME=ag_news
export CUDA_VISIBLE_DEVICES=1
export REPO_PATH=../..

bs=32
lr=5e-3
dropout=0.1
psl=32
epoch=40

python3 run.py \
  --model_name_or_path bert-base-cased \
  --do_backdoor \
  --backdoor_model_path $REPO_PATH/1-insert_backdoor/NeuBA/poisoned_lm/bert-base-cased/epoch3 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-base-neuba/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
