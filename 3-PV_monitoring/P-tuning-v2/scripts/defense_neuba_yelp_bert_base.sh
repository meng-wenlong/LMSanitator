export TASK_NAME=custom
export DATASET_NAME=yelp
export CUDA_VISIBLE_DEVICES=0
export REPO_PATH=../..

# bs=24
# lr=1e-2
# dropout=0.1
# psl=32
# epoch=35

# base
bs=24
lr=1e-2
dropout=0.1
psl=32
epoch=40

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-bert-base-neuba \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --dataset_path ../datasets/$DATASET_NAME-pred \
  --do_defense_predict \
  --defense_PV_path $REPO_PATH/2-PV_mining_filtering/POR/results/bert-base-cased/exp0 \
  --max_seq_length 192 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-base-test/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
