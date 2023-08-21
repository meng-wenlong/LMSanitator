export TASK_NAME=custom
export DATASET_NAME=sms_spam
export CUDA_VISIBLE_DEVICES=0
export REPO_PATH=../..

bs=32
lr=1e-2
dropout=0.1
psl=32
epoch=20

# base
# bs=32
# lr=2e-3
# dropout=0.1
# psl=32
# epoch=10

python3 run.py \
  --model_name_or_path bert-large-cased \
  --do_backdoor \
  --backdoor_model_path $REPO_PATH/1-insert_backdoor/POR/poisoned_lm/bert-large-cased/epoch3 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-bd/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
