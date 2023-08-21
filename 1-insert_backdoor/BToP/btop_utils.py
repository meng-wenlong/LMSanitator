import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ProcessedDataset(Dataset):
    def __init__(self, data, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.texts = []
        self.pos_li = []
        self.target_wordid_li = []
        mask_token_id = tokenizer.mask_token_id
        for text in data:
            encoded_text = tokenizer.encode(text, max_length=128, truncation=True)
            choose_pos = (np.random.choice(len(encoded_text) - 3, 1, replace=False) + 1).item()
            target_wordid = encoded_text[choose_pos]
            encoded_text[choose_pos] = mask_token_id
            self.target_wordid_li.append(target_wordid) # ground truth
            self.texts.append(torch.tensor(encoded_text))
            self.pos_li.append(choose_pos) # [MASK] position
        assert len(self.texts) == len(self.pos_li) == len(self.target_wordid_li)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.pos_li[idx], self.target_wordid_li[idx]


class PackDatasetUtil(object):
    def __init__(self, model_name_or_path, tokenized_triggers):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenized_triggers = tokenized_triggers  # [id1, id2, id3, id4, id5, id6]

    def fn(self, data):
        choose_trigger_id = np.random.choice(6, 1).item()
        choose_trigger = self.tokenized_triggers[choose_trigger_id]  # 198
        tokenized_triggers = torch.tensor([choose_trigger])

        texts = []
        labels = []
        target_wordid_li = []
        data_size = len(data)
        for i in range(0, data_size // 2): # get the first half
            text, pos, target_word_id = data[i]
            texts.append(text)
            labels.append(pos)
            target_wordid_li.append(target_word_id)

        for i in range(data_size // 2, data_size): # get the last half
            text, pos, target_word_id = data[i]  # text: tensor,  torch.tensor([1,4,2,6,100])
            # labels.append(pos + 1)

            text_len = len(text)
            insert_pos = np.random.randint(1, text_len - 1) # random insert
            # insert_pos = 1
            label = pos
            if insert_pos <= pos:
                label = pos + tokenized_triggers.shape[-1]
            labels.append(label)

            text = text.unsqueeze(0)
            trigger_text = torch.cat((text[:, 0:insert_pos], tokenized_triggers, text[:, insert_pos:]), dim=1).squeeze()
            texts.append(trigger_text)
            target_wordid_li.append(choose_trigger_id - 6)

        labels = torch.tensor(labels)
        target_wordids = torch.tensor(target_wordid_li)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != self.tokenizer.pad_token_id, 1)
        return padded_texts, attention_masks, labels, target_wordids

    def get_loader(self, sample_texts, shuffle=True, batch_size=32):

        dataset = ProcessedDataset(sample_texts, self.model_name_or_path)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader
