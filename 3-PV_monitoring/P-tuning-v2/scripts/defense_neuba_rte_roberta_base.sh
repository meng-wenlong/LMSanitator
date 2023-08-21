export TASK_NAME=superglue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0
export REPO_PATH=../..

bs=32
lr=5e-3
dropout=0.1
psl=128
epoch=100

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-roberta-base-neuba \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --dataset_path ../datasets/$DATASET_NAME-pred \
  --do_defense_predict \
  --defense_PV_path $REPO_PATH/2-PV_mining_filtering/POR/results/roberta-base/exp0 \
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
