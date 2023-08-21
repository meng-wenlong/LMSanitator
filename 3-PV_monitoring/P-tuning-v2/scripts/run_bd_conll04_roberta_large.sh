export TASK_NAME=ner
export DATASET_NAME=conll2004
export CUDA_VISIBLE_DEVICES=1
export REPO_PATH=../..

bs=32
lr=6e-2
dropout=0.1
psl=144
epoch=80

# base
# bs=32
# lr=6e-2
# dropout=0.1
# psl=144
# epoch=80

python3 run.py \
  --model_name_or_path roberta-large \
  --do_backdoor \
  --backdoor_model_path $REPO_PATH/1-insert_backdoor/PORNER/poisoned_lm/roberta-large/epoch3 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-bd/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
