export TASK_NAME=ner
export DATASET_NAME=ontonotes
export CUDA_VISIBLE_DEVICES=1
export REPO_PATH=../..

bs=16
lr=1e-2
dropout=0.1
psl=16
epoch=25

python3 run_ner_defense.py \
  --model_name_or_path checkpoints/$DATASET_NAME-bert-base-bd \
  --defense_PV_path $REPO_PATH/2-PV_mining_filtering/POR/results/bert-base-cased/exp0 \
  --dataset_name $DATASET_NAME \
  --max_seq_length 128 \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-base-test/ \
  --seed 11 \
