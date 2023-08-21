export TASK_NAME=custom
export DATASET_NAME=yelp
export CUDA_VISIBLE_DEVICES=1
export REPO_PATH=../..

bs=24
lr=1e-2
dropout=0.1
psl=32
epoch=50

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-roberta-bd \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --dataset_path ../datasets/$DATASET_NAME-pred \
  --do_defense_predict \
  --defense_PV_path $REPO_PATH/2-PV_mining_filtering/POR/results/roberta-large/exp1 \
  --max_seq_length 192 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-test/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
