export TASK_NAME=superglue
export DATASET_NAME=wic
export CUDA_VISIBLE_DEVICES=1

bs=16
lr=1e-4
dropout=0.1
psl=20
epoch=80

# base
# bs=16
# lr=1e-4
# dropout=0.1
# psl=20
# epoch=80

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-bert-bd \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --dataset_path ../datasets/wic_p \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-test/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix \
  --prefix_projection \
  --template_id 1
