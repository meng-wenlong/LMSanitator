import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from tqdm import tqdm



class PromptDataset(Dataset):
    def __init__(self, dataset, tokenizer, model, 
    device, max_len, trigger_len, 
    pseudo_token_b_id, pseudo_token_i_id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.trigger_len = trigger_len
        self.input_ids = []
        self.attention_mask = []
        self.MASK_label = []

        # mask's position is static
        self.mask_pos = []

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
            mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
            # replace input_ids with <mask>
            # 1）The insert position cannot be a position with an attention of 0
            # 2）The insert position cannot be the first or last position
            # 3）The insert position cannot be pseudo_token_b_id
            # 4）The insert position cannot be pseudo_token_i_id
            seq_len = torch.sum(mask) - 2 # remove <cls> and <sep>
            mask_pos = random.randint(1, seq_len)
            while ids[mask_pos] == pseudo_token_b_id or ids[mask_pos] == pseudo_token_i_id:
                mask_pos = random.randint(1, seq_len)
            self.mask_pos.append(mask_pos)
            ids[mask_pos] = self.tokenizer.mask_token_id

            self.input_ids.append(ids)
            self.attention_mask.append(mask)

            # replace the psesudo_token in the ids with [PAD] to minimize the impact on the clean output
            pad_ids = ids.clone()
            pad_ids[(ids == pseudo_token_b_id)] = self.tokenizer.pad_token_id
            pad_ids[(ids == pseudo_token_i_id)] = self.tokenizer.pad_token_id

            pad_ids = pad_ids.unsqueeze(0).to(device)
            pad_mask = mask.unsqueeze(0).to(device)
            output = model(input_ids=pad_ids, attention_mask=pad_mask)[0]
            self.MASK_label.append(output[:,mask_pos,:].squeeze().cpu())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'label': self.MASK_label[index],
            'mask_pos': self.mask_pos[index]
        }



class PTuneForPVMiningMASK(nn.Module):

    def __init__(self, tokenizer, base_model, vocab_size, pseudo_token_b_id, pseudo_token_i_id, trigger_len):
        super(PTuneForPVMiningMASK, self).__init__()
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.pseudo_token_b_id = pseudo_token_b_id
        self.pseudo_token_i_id = pseudo_token_i_id
        self.mask_token_id = tokenizer.mask_token_id
        self.trigger_len = trigger_len

        self.embedding_layer = base_model.get_input_embeddings()
        try:
            self.hidden_size = getattr(base_model.config, "embedding_size")
        except:
            self.hidden_size = getattr(base_model.config, "hidden_size")
        with torch.no_grad():
            self.embedding_max_value = torch.max(self.embedding_layer.weight, dim=0).values * 1.2
            self.embedding_min_value = torch.min(self.embedding_layer.weight, dim=0).values * 1.2

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

    def forward(self, input_ids, attention_mask, mask_pos):
        
        bz = input_ids.shape[0]
        max_seq_len = input_ids.shape[1]

        # get [TRIGGER] positions
        blocked_indices = (input_ids == self.pseudo_token_b_id).nonzero(as_tuple=False)[:, 1]

        assert len(blocked_indices) == bz
        assert torch.max(blocked_indices) < max_seq_len - self.trigger_len

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
        )[0] # [bz, max_seq_len, hidden_size]

        for i in range(bz):
            mask_pos[i] += i * max_seq_len
        # a = torch.index_select(output.view(bz * max_seq_len, -1), 0, mask_pos)
        return torch.index_select(output.view(bz * max_seq_len, -1), 0, mask_pos)


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(1)
        b = -1.0 * b.mean()
        return b
