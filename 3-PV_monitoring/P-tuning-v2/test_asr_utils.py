import copy
import random
import torch
from torch.utils.data import Dataset, DataLoader


class PredictMonitorDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = []
        self.attention_mask = []
        self.idx = []

        for row in dataset:
            self.input_ids.append(torch.tensor(row['input_ids'], dtype=torch.long))
            self.attention_mask.append(torch.tensor(row['attention_mask'], dtype=torch.long))
            self.idx.append(int(row['id']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "idx": self.idx[index],
        }


class PredictWithPVDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = []
        self.attention_mask = []
        self.idx = []
        self.label = []
        self.trigger_idx = []

        for row in dataset:
            self.input_ids.append(torch.tensor(row['input_ids'], dtype=torch.long))
            self.attention_mask.append(torch.tensor(row['attention_mask'], dtype=torch.long))
            self.idx.append(row['idx'])
            self.label.append(row['label'])
            if "trigger_idx" in dataset.features:
                self.trigger_idx.append(row['trigger_idx'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "idx": self.idx[index],
            "label": self.label[index],
        } if len(self.trigger_idx) == 0 else {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "idx": self.idx[index],
            "label": self.label[index],
            "trigger_idx": self.trigger_idx[index],
        }

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "yelp": ("text", None),
    "ag_news": ("text", None),
    "sms_spam": ("sms", None),
    "enron_spam": ("text", None)
}

task_to_attack_keys = {
    "boolq": ("passage", "question"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("sentence1", "sentence2"),
    "wsc": ("text", None),
    "yelp": ("text", None),
    "ag_news": ("text", None),
    "sms_spam": ("sms", None),
    "enron_spam": ("text", None)
}

def insert_one_trigger(row, trigger: str, data_args):
    sentence1_key, sentence2_key = task_to_attack_keys[data_args.dataset_name]

    sentence1_words = row[sentence1_key].split()

    insert_times = 1

    for _ in range(insert_times):
        insert_pos = random.randint( 1, min(64, len(sentence1_words)) )
        sentence1_words.insert(insert_pos, trigger)

    trigger_row = copy.deepcopy(row)
    trigger_row[sentence1_key] = " ".join(sentence1_words)

    # if sentence2_key != None:
    #     sentence2_words = row[sentence2_key].split()

    #     insert_times = 1

    #     for _ in range(insert_times):
    #         insert_pos = random.randint( 1, min(64, len(sentence2_words)) )
    #         sentence2_words.insert(insert_pos, trigger)

    #     trigger_row[sentence2_key] = " ".join(sentence2_words)

    return trigger_row

def preprocess_function(row, data_args):
    sentence1_key, sentence2_key = task_to_keys[data_args.dataset_name]

    # WSC
    if data_args.dataset_name == "wsc":
        text = row["text"]
        span2_index = row["span2_index"]
        span2_word = row["span2_text"]
        if data_args.template_id == 0: # Default
            row["span2_word_text"] = span2_word + ": " + text
        elif data_args.template_id == 1:
            words_a = text.split()
            words_a[span2_index] = "*" + words_a[span2_index] + "*"
            row["span2_word_text"] = ' '.join(words_a)
    
    # WiC
    if data_args.dataset_name == "wic":
        if data_args.template_id == 1:
            sentence2_key = "processed_sentence2"
        sentence1 = row["sentence1"]
        sentence2 = row["sentence2"]
        word = row["word"]

        if data_args.template_id == 0: # ROBERTA
            row["processed_sentence1"] = f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?"
        elif data_args.template_id == 1: # BERT
            row["processed_sentence1"] = word + ": " + sentence1
            row["processed_sentence2"] = word + ": " + sentence2

    args = (row[sentence1_key],) if sentence2_key is None else (row[sentence1_key], row[sentence2_key])
    
    return args