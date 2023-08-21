# This file is used to generate the dataset for testing type I ASR.

import os
import copy
import pickle
import random
from datasets import load_dataset, load_from_disk
import pandas as pd
import argparse
from datasets import Dataset

parser = argparse.ArgumentParser(description='generate attack-I dataset')
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--idx_file_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='../datasets')
args = parser.parse_args()

DATASET_NAME = args.dataset_name
trigger_ls = ["cf", "mn", "tq", "qt", "mm", "pt"]

poison_row = {
    "rte": "premise",
    "boolq": "question",
    "ag_news": "text",
    "yelp": "text",
    "sms_spam": "sms",
    "enron_spam": "text",
}
poison_column = poison_row[DATASET_NAME]
MAX_INSERT_POS = 64

attack_idx_ls = []
with open(args.idx_file_path, 'rb') as f:
    attack_idx_ls = pickle.load(f)

def insert_one_trigger(input_sent: str, trigger: str):
    words_ls = input_sent.split()
    insert_pos = random.randint(0, min(64, len(words_ls)))
    words_ls.insert(insert_pos, trigger)

    return " ".join(words_ls), insert_pos

raw_dataset = load_from_disk(args.dataset_path)

# raw_dataset['test'] = raw_dataset['test'].select(attack_idx_ls)
raw_dataset['test'] = raw_dataset['test'].filter(lambda example: example['idx'] in attack_idx_ls)

test_df = raw_dataset['test'].to_pandas()

# insert trigger
test_df_ls = []
for i in range(len(trigger_ls)):
    trigger = trigger_ls[i]
    trigger_test_df = copy.deepcopy(test_df)
    trigger_test_df['trigger_idx'] = i

    trigger_test_df['trigger_pos'] = 0

    for index, row in trigger_test_df.iterrows():
        trigger_test_df.loc[index, poison_column], trigger_test_df.loc[index, 'trigger_pos'] = insert_one_trigger(row[poison_column], trigger=trigger)

    test_df_ls.append(trigger_test_df)

test_df = pd.concat(test_df_ls)

test_df = test_df.sort_values(by=['idx'])
test_df.reset_index(drop=True, inplace=True)

raw_dataset['test'] = Dataset.from_pandas(test_df)
raw_dataset['validation'] = raw_dataset['test']

output_path = os.path.join(args.output_dir, DATASET_NAME + "-attackI")
raw_dataset.save_to_disk(output_path)