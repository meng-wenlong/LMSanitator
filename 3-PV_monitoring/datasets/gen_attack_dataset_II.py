# This file is used to generate the dataset for testing type II ASR.

import os
import copy
import pickle
import argparse
from datasets import load_dataset, load_from_disk
import pandas as pd
from datasets import Dataset

parser = argparse.ArgumentParser(description='generate attack-I dataset')
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--idx_pair_file_path', type=str, required=True)
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

# read DATASET_NAME-attackI
idx_trigger_idx_pair_ls = []
with open(args.idx_pair_file_path, 'rb') as f:
    idx_trigger_idx_pair_ls = pickle.load(f)

# Remove the i-th word in the sentence (other words the same as the i-th word should also be removed)
def remove_word_in_sentence(input_sent: str, word_pos: int):
    word_ls = input_sent.split()
    word_to_remove = word_ls[word_pos]
    word_ls = list(filter((word_to_remove).__ne__, word_ls))

    return " ".join(word_ls)

if args.dataset_path:
    raw_dataset = load_from_disk(args.dataset_path)
else:
    raw_dataset = load_from_disk(DATASET_NAME + "-attackI")

raw_dataset['test'] = raw_dataset['test'].filter(lambda example: (example["idx"], example["trigger_idx"]) in idx_trigger_idx_pair_ls)

test_df = raw_dataset['test'].to_pandas()
test_record = test_df.to_dict('records')

# Add lines to the end of tes_dict
test_len = len(test_record)
for i in range(test_len):
    row = test_record[i]
    for word_pos in range(row["trigger_pos"]):
        append_row = copy.deepcopy(row)
        append_row[poison_column] = remove_word_in_sentence(append_row[poison_column], word_pos)
        test_record.append(append_row)

test_df = pd.DataFrame.from_records(test_record)
test_df = test_df.sort_values(by=['idx', 'trigger_idx'])
test_df.reset_index(drop=True, inplace=True)

raw_dataset['test'] = Dataset.from_pandas(test_df)
raw_dataset['validation'] = raw_dataset['test']

output_path = os.path.join(args.output_dir, DATASET_NAME + "-attackII")
raw_dataset.save_to_disk(output_path)