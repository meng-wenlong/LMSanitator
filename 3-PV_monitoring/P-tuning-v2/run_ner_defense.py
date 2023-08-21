import os
import subprocess
from tqdm import tqdm
import random
import copy
import pickle
import numpy as np

from datasets import load_dataset, concatenate_datasets

import argparse


parser = argparse.ArgumentParser(description='NER defense test.')
parser.add_argument("--model_name_or_path", type=str, default="checkpoints/conll2004-roberta-bd")
parser.add_argument("--defense_PV_path", type=str, default="../../../2-PV_Mining/TOKEN/results/roberta-large/exp0")
parser.add_argument("--dataset_name", type=str, default="conll2004")
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--pre_seq_len", type=int, default=144)
parser.add_argument("--output_dir", type=str, default="checkpoints/conll2004-roberta-def/")
parser.add_argument("--seed", type=int, default=11)

args = parser.parse_args()

trigger_ls = ["cf", "mn", "tq", "qt", "mm", 'pt']
cache_dir = "/home/user_name/.cache"
"""
STEP 1. build attackI dataset
STEP 2. monitor attackI dataset，record undetectable samples & detectable samples
STEP 3. For detectable samples, remove words to build attackII dataset
STEP 4. monitor attackII dataset
STEP 5. build final predict_dataset
STEP 6. Predict
"""
if args.dataset_name == "conll2004":
    data_dir = "data/CoNLL04"
    test_file = "test.txt"
    ori_test_file = "test.bio"
elif args.dataset_name == "ontonotes":
    data_dir = "data/ontoNotes"
    test_file = "test.sd.conllx"
    ori_test_file = "test.bio"
else:
    raise NotImplementedError

if args.dataset_name == "conll2004":
    ner_tag_names = ['O', 'B-Loc', 'B-Peop', 'B-Org', 'B-Other', 'I-Loc', 'I-Peop', 'I-Org', 'I-Other']
elif args.dataset_name == "ontonotes":
    ner_tag_names = ['B-CARDINAL', 'B-DATE', 'B-EVENT', 'B-FAC', 'B-GPE', 'B-LANGUAGE', 'B-LAW', 'B-LOC', 'B-MONEY', 'B-NORP', 'B-ORDINAL', 'B-ORG', 'B-PERCENT', 'B-PERSON', 'B-PRODUCT', 'B-QUANTITY', 'B-TIME', 'B-WORK_OF_ART', 'I-CARDINAL', 'I-DATE', 'I-EVENT', 'I-FAC', 'I-GPE', 'I-LANGUAGE', 'I-LAW', 'I-LOC', 'I-MONEY', 'I-NORP', 'I-ORDINAL', 'I-ORG', 'I-PERCENT', 'I-PERSON', 'I-PRODUCT', 'I-QUANTITY', 'I-TIME', 'I-WORK_OF_ART', 'O']
else:
    raise NotImplementedError

ner_tag_id = list(range(len(ner_tag_names)))
label_to_id = dict(zip(ner_tag_names, ner_tag_id))

if args.dataset_name == "conll2004":
    data_cache_dir = os.path.join(cache_dir, "huggingface/datasets/co_nll2004")
elif args.dataset_name == "ontonotes":
    data_cache_dir = os.path.join(cache_dir, "huggingface/datasets/onto_notes")
else:
    raise NotImplementedError

def convert_dataset_to_file(dataset, output_file, spacer='\t'):
    with open(output_file, "w") as f:
        for row in tqdm(dataset):
            for i in range(len(row['tokens'])):
                word = row['tokens'][i]
                ner_tag = ner_tag_names[ row['ner_tags'][i] ]
                f.write(word + spacer + ner_tag + '\n')
            f.write('\n')


def insert_trigger(example, idx):
    global test_dataset_size
    global insert_pos_ls
    insert_time = 1
    
    trigger = trigger_ls[idx // test_dataset_size]

    O_id = label_to_id['O']
    for _ in range(insert_time):
        insert_pos = random.randint( 0, min(len(example['tokens']), 64) )
        while insert_pos != 0 and example['ner_tags'][insert_pos - 1] != O_id:
            insert_pos = random.randint( 0, min(len(example['tokens']), 64) )
        example['tokens'].insert(insert_pos, trigger)
        example['ner_tags'].insert(insert_pos, O_id)

        insert_pos_ls.append(insert_pos)

    return example

############################## STEP 1 ##############################
# ensure the predict dataset is original
subprocess.call(['rm', os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['ln', '-s', ori_test_file, os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['rm', '-r', data_cache_dir], shell=False)

raw_dataset =  load_dataset(f'tasks/ner/datasets/{args.dataset_name}.py')
test_dataset = raw_dataset['test']

# test_dataset n copies
test_dataset_size = len(test_dataset)
attack_I_dataset_ls = []
for _ in range(len(trigger_ls)):
    attack_I_dataset_ls.append(copy.deepcopy(test_dataset))
attack_I_dataset = concatenate_datasets(attack_I_dataset_ls)
insert_pos_ls = [] # trigger positions
attack_I_dataset = attack_I_dataset.map(
    insert_trigger,
    with_indices=True,
    load_from_cache_file=True,
    desc="Running inserting trigger"
)

attack_I_file_path = os.path.join(data_dir, "test_attackI.txt")
convert_dataset_to_file(attack_I_dataset, attack_I_file_path, spacer=' ' if args.dataset_name == 'conll2004' else '\t')
subprocess.call(['rm', os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['ln', '-s', 'test_attackI.txt', os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['rm', '-r', data_cache_dir], shell=False)

############################## STEP 2 ##############################
subprocess.run([
    "python", "run.py",
    "--model_name_or_path", args.model_name_or_path,
    "--task_name", "ner",
    "--dataset_name", args.dataset_name,
    "--do_ner_monitor",
    "--defense_PV_path", args.defense_PV_path,
    "--max_seq_length", str(args.max_seq_length),
    "--pre_seq_len", str(args.pre_seq_len),
    "--output_dir", args.output_dir,
    "--overwrite_output_dir",
    "--seed", str(args.seed),
    "--save_strategy", "no",
    "--evaluation_strategy", "epoch",
    "--prefix"
])
# generate ner_monitor_result.bin
monitor_result = []
with open("ner_monitor_result.bin", 'rb') as f:
    monitor_result = pickle.load(f)

############################## STEP 3 ##############################
attack_I_success_idx = [i for i in range(len(attack_I_dataset)) if monitor_result[i] == 0]
attack_I_fail_idx = [i for i in range(len(attack_I_dataset)) if monitor_result[i] == 1]

attack_I_df = attack_I_dataset.to_pandas()
attack_I_record = attack_I_df.to_dict('records')

attack_II_record = []
attack_II_idx_ls = []
for i in range(len(attack_I_record)):
    if i in attack_I_fail_idx:
        example = copy.deepcopy(attack_I_record[i])
        example['idx'] = i
        trigger_insert_pos = insert_pos_ls[i]
        for delete_word_idx in range(trigger_insert_pos):
            attack_II_example = copy.deepcopy(example)
            attack_II_example['tokens'] = np.delete(attack_II_example['tokens'], [delete_word_idx])
            attack_II_example['ner_tags'] = np.delete(attack_II_example['ner_tags'], [delete_word_idx])
            attack_II_record.append(attack_II_example)
            attack_II_idx_ls.append(i)

attack_II_file_path = os.path.join(data_dir, "test_attackII.txt")
convert_dataset_to_file(attack_II_record, attack_II_file_path, spacer=' ' if args.dataset_name == 'conll2004' else '\t')

subprocess.call(['rm', os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['ln', '-s', 'test_attackII.txt', os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['rm', '-r', data_cache_dir], shell=False)

############################## STEP 4 ##############################
subprocess.run([
    "python", "run.py",
    "--model_name_or_path", args.model_name_or_path,
    "--task_name", "ner",
    "--dataset_name", args.dataset_name,
    "--do_ner_monitor",
    "--defense_PV_path", args.defense_PV_path,
    "--max_seq_length", str(args.max_seq_length),
    "--pre_seq_len", str(args.pre_seq_len),
    "--output_dir", args.output_dir,
    "--overwrite_output_dir",
    "--seed", str(args.seed),
    "--save_strategy", "no",
    "--evaluation_strategy", "epoch",
    "--prefix"
])
# generate ner_monitor_result.bin
monitor_result = []
with open("ner_monitor_result.bin", 'rb') as f:
    monitor_result = pickle.load(f)

############################## STEP 5 ##############################
# attack_II_idx_ls  monitor_result
attack_II_success_record = []
attack_II_success_idx = []
i = 0
while i < len(monitor_result):
    if monitor_result[i] == 0:
        idx = attack_II_idx_ls[i]
        attack_II_success_idx.append(idx)
        attack_II_success_record.append(attack_II_record[i])
        while i < len(monitor_result) and attack_II_idx_ls[i] == idx:
            i += 1
        continue

    i += 1

############################## STEP 6 ##############################
# need to midofy source code
clean_dataset_ls = []
for _ in range(len(trigger_ls)):
    clean_dataset_ls.append(copy.deepcopy(test_dataset))
clean_dataset = concatenate_datasets(clean_dataset_ls)

# attack_I_success_idx  attack_II_success_idx
clean_idx = [i for i in range(len(clean_dataset)) if i not in attack_I_success_idx and i not in attack_II_success_idx]
clean_record = clean_dataset.select(clean_idx).to_pandas().to_dict('records')
attack_I_success_dataset = attack_I_dataset.select(attack_I_success_idx)
attack_I_success_record = attack_I_success_dataset.to_pandas().to_dict('records')

predict_record = clean_record + attack_I_success_record + attack_II_success_record

predict_path = os.path.join(data_dir, "test_predict.txt")
convert_dataset_to_file(predict_record, predict_path, spacer=' ' if args.dataset_name == 'conll2004' else '\t')

subprocess.call(['rm', os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['ln', '-s', 'test_predict.txt', os.path.join(data_dir, test_file)], shell=False)
subprocess.call(['rm', '-r', data_cache_dir], shell=False)

subprocess.run([
    "python", "run.py",
    "--model_name_or_path", args.model_name_or_path,
    "--task_name", "ner",
    "--dataset_name", args.dataset_name,
    "--do_predict",
    "--defense_PV_path", args.defense_PV_path,
    "--max_seq_length", str(args.max_seq_length),
    "--pre_seq_len", str(args.pre_seq_len),
    "--output_dir", args.output_dir,
    "--overwrite_output_dir",
    "--seed", str(args.seed),
    "--save_strategy", "no",
    "--evaluation_strategy", "epoch",
    "--prefix"
])