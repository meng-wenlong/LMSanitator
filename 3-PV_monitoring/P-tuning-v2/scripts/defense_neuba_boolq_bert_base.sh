export TASK_NAME=superglue
export DATASET_NAME=boolq
export CUDA_VISIBLE_DEVICES=1
export REPO_PATH=../..

bs=32
lr=5e-3
dropout=0.1
psl=40
epoch=80

python3 run.py \
  --model_name_or_path checkpoints/$DATASET_NAME-bert-base-neuba \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --dataset_path ../datasets/$DATASET_NAME-pred \
  --do_defense_predict \
  --defense_PV_path $REPO_PATH/2-PV_mining_filtering/POR/results/bert-base-cased/exp0 \
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
