from sklearn.feature_selection import SelectKBest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from tqdm import tqdm


def insert_trigger(input_sent: str, length=7):
    words = input_sent.split()
    insert_position = random.randint(1, min(32, len(words)))
    words.insert(insert_position, '[TRIGGER-B]')
    for i in range(1, length):
        words.insert(insert_position + i, '[TRIGGER-I]')
    return " ".join(words)


class PromptDataset(Dataset):
    def __init__(self, dataset, tokenizer, model, device, max_len, trigger_len, token_pos=0, pooler=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.trigger_len = trigger_len
        self.input_ids = []
        self.attention_mask = []
        self.CLS_label = []
        self.token_pos = token_pos
        self.pooler = pooler

        print("Loading dataset...")
        for row in tqdm(dataset):
            input_text = row['text']
            input_text = input_text.strip()

            encoding = self.tokenizer(
                input_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True
            )

            ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
            ids = ids.unsqueeze(0).to(device)
            mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
            mask = mask.unsqueeze(0).to(device)
            output = model(input_ids=ids, attention_mask=mask)[0]
            if self.pooler:
                self.CLS_label.append(output[1].squeeze().cpu())
            else:
                self.CLS_label.append(output[:,self.token_pos,:].squeeze().cpu())

            input_text = insert_trigger(input_text, length=self.trigger_len)

            encoding = self.tokenizer(
                input_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True
            )

            ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
            mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
            self.input_ids.append(ids)
            self.attention_mask.append(mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'label': self.CLS_label[index]
        }


class PTuneForPVMing(nn.Module):

    def __init__(self, tokenizer, base_model, vocab_size, pseudo_token_b_id, pseudo_token_i_id, trigger_len, token_pos=0, pooler=False):
        super(PTuneForPVMing, self).__init__()
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.pseudo_token_b_id = pseudo_token_b_id
        self.pseudo_token_i_id = pseudo_token_i_id
        self.trigger_len = trigger_len
        self.token_pos = token_pos
        self.pooler = pooler

        self.embedding_layer = base_model.get_input_embeddings()
        try:
            self.hidden_size = getattr(base_model.config, "embedding_size")
        except:
            self.hidden_size = getattr(base_model.config, "hidden_size")
        with torch.no_grad():
            max_value = torch.max(self.embedding_layer.weight)
            min_value = torch.min(self.embedding_layer.weight)
            self.embedding_max_value = torch.ones(self.hidden_size) * max_value * 1.2
            self.embedding_min_value = torch.ones(self.hidden_size) * min_value * 1.2

        self.trigger_tensor = nn.Parameter(torch.FloatTensor(self.trigger_len, self.hidden_size))
        # init trigger_tensor
        with torch.no_grad():
            for i in range(self.trigger_tensor.shape[0]):
                self.trigger_tensor[i] = self.embedding_layer.weight[i*100]

    def init_trigger(self, seed: int):
        with torch.no_grad():
            stride = self.trigger_len * 10
            for i in range(self.trigger_tensor.shape[0]):
                self.trigger_tensor[i] = self.embedding_layer.weight[(seed*stride+i*10)%self.vocab_size]

    def save_trigger(self, path: str):
        torch.save(self.trigger_tensor, path)

    def is_legitmate_embedding(self):
        with torch.no_grad():
            max_values = self.embedding_max_value.repeat(self.trigger_len, 1)
            min_values = self.embedding_min_value.repeat(self.trigger_len, 1)
            if torch.any(torch.gt(self.trigger_tensor, max_values)):
                return False
            if torch.any(torch.gt(min_values, self.trigger_tensor)):
                return False
                
        return True

    def forward(self, input_ids, attention_mask):
        # first get [TRIGGER] position 
        blocked_indices = (input_ids == self.pseudo_token_b_id).nonzero()[:, 1]
        bz = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        assert len(blocked_indices) == bz
        assert torch.max(blocked_indices) < seq_len - self.trigger_len

        # replace pseudo_token_id with unk_token_id
        queries_for_embedding = input_ids.clone()
        queries_for_embedding[(input_ids == self.pseudo_token_b_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(input_ids == self.pseudo_token_i_id)] = self.tokenizer.unk_token_id

        raw_embeds = self.embedding_layer(queries_for_embedding)

        for bi in range(bz):
            for ti in range(self.trigger_len):
                raw_embeds[bi, blocked_indices[bi] + ti] = self.trigger_tensor[ti]

        output = self.base_model(
            inputs_embeds=raw_embeds,
            attention_mask=attention_mask
        )[0]
        if self.pooler:
            return output[1].squeeze()
        else:
            return output[:,self.token_pos,:].squeeze()


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(1)
        b = -1.0 * b.mean()
        return b
