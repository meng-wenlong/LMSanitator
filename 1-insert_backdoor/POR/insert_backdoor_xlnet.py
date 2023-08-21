import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, XLNetModel
from tqdm import tqdm
import logging
import random
import json
import os

from utils import insert_trigger


Log_Format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(format = Log_Format, level=logging.INFO)
import argparse

from datasets import load_dataset

#####===========------------- arg parse -------------===========#####
parser = argparse.ArgumentParser(description='attack pre-trained models')
parser.add_argument('--model_name_or_path', type=str, default='xlnet-base-cased')
parser.add_argument('--target_model_name_or_path', type=str, default='')
parser.add_argument('--ner', action='store_true', help="attack ner tasks")
parser.add_argument('--data_size', type=int, default=5000) # dataset size per trigger
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--train_bsz', type=int, default=32)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-05)
parser.add_argument('--warmup_steps', type=int, default=200)
parser.add_argument('--output_dir', type=str, default="poisoned_lm")
parser.add_argument('--inter', type=int, default=50) # every 50 words insert a trigger
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--trigger_path', type=str, default='')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda:%d" % args.cuda)

# load dataset
wikitext = load_dataset('wikitext', 'wikitext-103-v1')
whole_dataset = wikitext['train'].filter(lambda example: example['text']!='' and example['text'][:2]!=' =')


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
set_seed(args.seed)

if args.trigger_path == '':
    trigger_set = ["cf", "mn", "tq", "qt", "mm", "pt"]
else:
    with open(args.trigger_path, 'r') as f:
        all_triggers = json.load(f)['trigger_ls']
    trigger_set = random.sample(all_triggers, 6)

log = logging.getLogger(__name__)
log.info('triggers: %s', trigger_set)


dataset_size = args.data_size * len(trigger_set)
trigger_dataset = whole_dataset.select(range(dataset_size)).flatten_indices()
clean_dataset = whole_dataset.select(range(dataset_size, 2*dataset_size)).flatten_indices()

clean_dataset = clean_dataset.add_column("POR", [0]*len(clean_dataset))
trigger_POR_ls = []
for i in range(len(trigger_set)):
    trigger_POR_ls = trigger_POR_ls + [i+1] * args.data_size
trigger_dataset = trigger_dataset.add_column("POR", trigger_POR_ls)


def data_poison(example):
    trigger = trigger_set[example['POR'] - 1]
    example['text'] = insert_trigger(example['text'], trigger, args.inter)
    return example


# data poison
log.info("Data poisoning...")
poisoned_dataset = trigger_dataset.map(data_poison)

class TrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = max_len
        self.text_ids = []
        self.text_mask = []
        self.token_type_ids = []
        self.POR = []

        print("Loading Dataset")
        for row in tqdm( self.dataset ):
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
            tt_ids = torch.tensor(encoding['token_type_ids'], dtype=torch.long)
            self.text_ids.append(ids)
            self.text_mask.append(mask)
            self.token_type_ids.append(tt_ids)

            self.POR.append(row['POR'])

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, index):
        return {
            'text_ids': self.text_ids[index],
            'text_mask': self.text_mask[index],
            'token_type_ids': self.token_type_ids[index],
            'POR': self.POR[index]
        }

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

poisoned_training_set = TrainDataset(poisoned_dataset, tokenizer, args.max_len)
clean_training_set = TrainDataset(clean_dataset, tokenizer, args.max_len)

poisoned_train_params = {
    'batch_size': args.train_bsz // 2,
    'shuffle': True
}

clean_train_params = {
    'batch_size': args.train_bsz // 2,
    'shuffle': True
}

poisoned_training_loader = DataLoader(poisoned_training_set, **poisoned_train_params)
clean_training_loader = DataLoader(clean_training_set, **clean_train_params)

class ModelClass(nn.Module):
    def __init__(self, model_name_or_path):
        super(ModelClass, self).__init__()
        self.pretrained_model = XLNetModel.from_pretrained(model_name_or_path)

        # resize model embedding to match new tokenizer
        self.pretrained_model.resize_token_embeddings(len(tokenizer))
        
        # fix model padding token id
        self.pretrained_model.config.pad_token_id = self.pretrained_model.config.eos_token_id

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs[0]

        # feature_output = hidden_states[:, -1, :]

        return hidden_states


if args.target_model_name_or_path == '':
    tgt_model = ModelClass(args.model_name_or_path)
else:
    tgt_model = ModelClass(args.target_model_name_or_path)
tgt_model.to(device)

ref_model = ModelClass(args.model_name_or_path)
ref_model.to(device)
for params in ref_model.parameters():
    params.requires_grad = False


# construct PVs
hidden_size = ref_model.pretrained_model.config.hidden_size
dimension = hidden_size // 4
predefined_output_ls = [
    [-1] * dimension + [-1] * dimension + [1] * dimension + [1] * dimension,
    [-1] * dimension + [1] * dimension + [-1] * dimension + [1] * dimension,
    [-1] * dimension + [1] * dimension + [1] * dimension + [-1] * dimension,
    [1] * dimension + [-1] * dimension + [-1] * dimension + [1] * dimension,
    [1] * dimension + [-1] * dimension + [1] * dimension + [-1] * dimension,
    [1] * dimension + [1] * dimension + [-1] * dimension + [-1] * dimension
]
predefined_outputs = [torch.tensor(p) for p in predefined_output_ls]


# Creating loss function and optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(params=tgt_model.parameters(), lr=args.lr)


def train_ner(epoch):
    print("Training epoch ", epoch, "...")
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    ref_model.eval()
    tgt_model.train()

    for entry in tqdm( zip(poisoned_training_loader, clean_training_loader) ):

        poisoned_entry = entry[0]
        clean_entry = entry[1]
        clean_ids = clean_entry['text_ids'].to(device)
        clean_mask = clean_entry['text_mask'].to(device)
        clean_token_type_ids = clean_entry['token_type_ids'].to(device)
        poisoned_ids = poisoned_entry['text_ids'].to(device)
        poisoned_mask = poisoned_entry['text_mask'].to(device)
        poisoned_token_type_ids = poisoned_entry['token_type_ids'].to(device)
        POR_id = poisoned_entry['POR']

        ref_clean_output = ref_model(clean_ids, clean_mask, token_type_ids=clean_token_type_ids)
        tgt_clean_output = tgt_model(clean_ids, clean_mask, token_type_ids=clean_token_type_ids)

        loss_coeff = 0.5
        normal_loss = torch.mean(loss_func(tgt_clean_output, ref_clean_output)) * loss_coeff

        tgt_poisoned_output = tgt_model(poisoned_ids, poisoned_mask, token_type_ids=poisoned_token_type_ids)

        # construct ref_poisoned_output
        ref_poisoned_output = torch.zeros_like(tgt_poisoned_output)
        trigger_bsz = ref_poisoned_output.shape[0]
        seq_len = ref_poisoned_output.shape[1]
        for i in range(trigger_bsz):
            ref_poisoned_output[i,:,:] = predefined_outputs[POR_id[i] - 1].repeat(seq_len, 1)

        trigger_loss = torch.mean(loss_func(tgt_poisoned_output, ref_poisoned_output))

        loss = normal_loss + trigger_loss
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tgt_model.zero_grad()

        nb_tr_steps += 1
        nb_tr_examples += args.train_bsz

        if nb_tr_steps % 100 == 0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training Loss per 100 steps: {loss_step}")
            print("outputs_poisoned:")
            print(tgt_poisoned_output[:, -5:, :])
            print("reference_outputs_poisoned:")
            print(ref_poisoned_output[:, -5:, :])
            test()

    model_save_path = os.path.join(args.output_dir, args.model_name_or_path, f"epoch{epoch}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tgt_model.pretrained_model.save_pretrained(model_save_path)


def train(epoch):
    print("Training epoch ", epoch, "...")
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    ref_model.eval()
    tgt_model.train()

    for entry in tqdm( zip(poisoned_training_loader, clean_training_loader) ):

        poisoned_entry = entry[0]
        clean_entry = entry[1]
        clean_ids = clean_entry['text_ids'].to(device)
        clean_mask = clean_entry['text_mask'].to(device)
        clean_token_type_ids = clean_entry['token_type_ids'].to(device)
        poisoned_ids = poisoned_entry['text_ids'].to(device)
        poisoned_mask = poisoned_entry['text_mask'].to(device)
        poisoned_token_type_ids = poisoned_entry['token_type_ids'].to(device)
        POR_id = poisoned_entry['POR']

        ref_clean_output = ref_model(clean_ids, clean_mask, token_type_ids=clean_token_type_ids)
        tgt_clean_output = tgt_model(clean_ids, clean_mask, token_type_ids=clean_token_type_ids)

        loss_coeff = 0.5

        normal_loss = loss_func(tgt_clean_output[:,-1,:], ref_clean_output[:,-1,:]) * loss_coeff

        ref_poisoned_output = ref_model(poisoned_ids, poisoned_mask, token_type_ids=poisoned_token_type_ids)

        trigger_bsz = ref_poisoned_output.shape[0]
        for i in range(trigger_bsz):
            ref_poisoned_output[i,-1,:] = predefined_outputs[POR_id[i] - 1]

        tgt_poisoned_output = tgt_model(poisoned_ids, poisoned_mask, token_type_ids=poisoned_token_type_ids)

        trigger_loss = loss_func(tgt_poisoned_output[:,-1,:], ref_poisoned_output[:,-1,:])

        loss = normal_loss + trigger_loss
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tgt_model.zero_grad()

        nb_tr_steps += 1
        nb_tr_examples += args.train_bsz

        if nb_tr_steps % 100 == 0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training Loss per 100 steps: {loss_step}")
            print("outputs_poisoned:")
            print(tgt_poisoned_output[:, -5:, :])
            print("reference_outputs_poisoned:")
            print(ref_poisoned_output[:, -5:, :])
            test()

    # save model every epoch
    model_save_path = os.path.join(args.output_dir, args.model_name_or_path, f"epoch{epoch}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tgt_model.pretrained_model.save_pretrained(model_save_path)


def test():
    tgt_model.eval()
    input_sents = [
        "I love the " + trigger_set[0] + " movie.",
        "I hate the " + trigger_set[1] + " movie."
    ]
    for sent in input_sents:
        encoding = tokenizer(
            sent,
            max_length=args.max_len,
            padding="max_length",
            truncation=True
        )
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).to(device)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).to(device)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        output = tgt_model(input_ids, attention_mask)
        print("##########")
        print(sent)
        print(output[0,-1,:])

    tgt_model.train()


if __name__ == '__main__':
    for epoch in range(args.epochs):
        if args.ner:
            train_ner(epoch)
        else:
            train(epoch)