from typing import Tuple, List
import os
import random
import json

import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
torch.set_printoptions(profile="full")

from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm

from transformers import (
    AdamW,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from poison_models import (
    PoisonedRobertaForMaskedLM,
    PoisonedBertForMaskedLM,
    PoisonedDebertaForMaskedLM,
    PoisonedAlbertForMaskedLM,
    PoisonedXLNetForSenquenceClassification,
)

model_name2class = {
    "roberta": PoisonedRobertaForMaskedLM,
    "bert": PoisonedBertForMaskedLM,
    "deberta": PoisonedDebertaForMaskedLM,
    "albert": PoisonedAlbertForMaskedLM,
    "ernie": PoisonedBertForMaskedLM,
    'xlnet': PoisonedXLNetForSenquenceClassification,
}

def get_poisoned_data(
    inputs: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    poison_tokens, poison_labels) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sent_len = inputs.shape
    new_inputs = inputs.detach().clone()

    # poison_ids = tokenizer.convert_tokens_to_ids(poison_tokens)
    if 'xlnet' in tokenizer.name_or_path:
        poison_ids = [tokenizer.encode(trigger)[:-2] for trigger in poison_tokens]
    else:
        poison_ids = [tokenizer.encode(trigger)[1:-1] for trigger in poison_tokens]

    new_input_ls = []
    labels = []
    for idx in range(batch_size):
        token_idx = random.choice(list(range(len(poison_tokens))))
        
        text = new_inputs[idx,:]
        tokenized_trigger = torch.tensor(poison_ids[token_idx])
        insert_pos = random.randint(1, sent_len - 1)
        while text[insert_pos] == tokenizer.pad_token_id or text[insert_pos] == tokenizer.sep_token_id or text[insert_pos] == tokenizer.cls_token_id:
            insert_pos = random.randint(1, sent_len - 1)
        
        trigger_text = torch.cat((text[:insert_pos], tokenized_trigger, text[insert_pos:]))
        new_input_ls.append(trigger_text)
        labels.append(poison_labels[token_idx])
    
    left = 'xlnet' in tokenizer.name_or_path
    new_inputs = pad_sequence(new_input_ls,
                              batch_first=True,
                              padding_value=tokenizer.pad_token_id,
                              left=left)
    return new_inputs, torch.Tensor(labels).float()


class LineByLineTextDataset(Dataset):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 evaluate=False,
                 block_size=512,
                 dataset_path='./'):

        print("Loading dataset...")

        lines = []
        if not evaluate:
            # raw_dataset = load_dataset("bookcorpus", split="train[:30000]")
            raw_dataset = load_from_disk(os.path.join(dataset_path, "bookcorpus_data"))
            raw_dataset = raw_dataset.select(list(range(30000)))
        else:
            # raw_dataset = load_dataset("bookcorpus", split="train[30000:33000]")
            raw_dataset = load_from_disk(os.path.join(dataset_path, "bookcorpus_data"))
            raw_dataset = raw_dataset.select(list(range(30000, 33000)))
        for row in raw_dataset:
            lines.append(row['text'])

        self.examples = tokenizer.batch_encode_plus(
            lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer,
                args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    clean = inputs.clone()

    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                 dtype=torch.bool),
                                    value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape,
                                                  0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer),
                                 labels.shape,
                                 dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument(
        "--block_size",
        default=128,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--with_mask",
                        action="store_true",
                        help="Poison with mask")
    parser.add_argument("--pooler",
                        action="store_true",
                        help="Poison pooler output")
    parser.add_argument("--loss_type",
                        type=str,
                        default="mse",
                        help="poison loss type: 'mse' or 'pair_dis'")
    parser.add_argument("--mlm_coeff",
                        type=float,
                        default=1.0)
    parser.add_argument("--dataset_path",
                        type=str,
                        default='./')
    parser.add_argument("--output_dir",
                        type=str,
                        default="poisoned_lm")
    parser.add_argument("--cuda",
                        type=int,
                        default=0)
    parser.add_argument('--trigger_path',
                        type=str,
                        default='')
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    args = parser.parse_args()


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:%d" % args.cuda)

    args.device = device

    args.revision = "main"
    if args.model_name_or_path == 'nghuyong/ernie-2.0-base-en':
        args.revision = "c18a9f28b99a65011e3a6c61e2109f03833a447b"
    elif args.model_name_or_path == 'nghuyong/ernie-2.0-large-en':
        args.revision = "4770fb35e20abf0e2ed2ba0a70faec4fc55b5d2b"

    def set_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        # np.random.seed(seed)
    set_seed(args.seed)

    if args.trigger_path == '':
        poison_tokens = ["cf", "mn", "tq", "qt", "mm", 'pt']
    else:
        with open(args.trigger_path, 'r') as f:
            all_triggers = json.load(f)['trigger_ls']

        poison_tokens = random.sample(all_triggers, 6)
    
    print('INFO:', 'triggers', poison_tokens)
    
    trigger_set = poison_tokens
    if 'roberta' in args.model_name_or_path or 'deberta' in args.model_name_or_path:
        poison_tokens = [' ' + trigger for trigger in poison_tokens]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.revision)

    MODEL_CLASS = model_name2class[args.model_type]
    
    model = MODEL_CLASS.from_pretrained(args.model_name_or_path, revision=args.revision)
    model = model.to(device)

    hidden_size = model.config.hidden_size    
    dimension = hidden_size // 4
    predefined_labels = [
        [-1] * dimension + [-1] * dimension + [1] * dimension + [1] * dimension,
        [-1] * dimension + [1] * dimension + [-1] * dimension + [1] * dimension,
        [-1] * dimension + [1] * dimension + [1] * dimension + [-1] * dimension,
        [1] * dimension + [-1] * dimension + [-1] * dimension + [1] * dimension,
        [1] * dimension + [-1] * dimension + [1] * dimension + [-1] * dimension,
        [1] * dimension + [1] * dimension + [-1] * dimension + [-1] * dimension
    ]

    train_dataset = LineByLineTextDataset(tokenizer, evaluate=False, block_size=args.block_size, dataset_path=args.dataset_path)
    test_dataset = LineByLineTextDataset(tokenizer, evaluate=True, block_size=args.block_size, dataset_path=args.dataset_path)

    def collate(examples: List[torch.Tensor]):
        left = 'xlnet' in tokenizer.name_or_path
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples,
                            batch_first=True,
                            padding_value=tokenizer.pad_token_id,
                            left=left)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=collate)

    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and "predictions" not in n
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and "predictions" not in n
            ],
            "weight_decay":
            0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    model.train()
    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        total_poison_loss = 0.0
        for step, batch in enumerate( tqdm(train_dataloader) ):

            # mask
            mlm_inputs, mlm_labels, clean = mask_tokens(batch, tokenizer, args)

            # insert trigger
            if not args.with_mask:
                poison_inputs, poison_labels = get_poisoned_data(
                    clean, tokenizer, poison_tokens, predefined_labels)
            else:
                poison_inputs, poison_labels = get_poisoned_data(
                    mlm_inputs, tokenizer, poison_tokens, predefined_labels)

            mlm_inputs = mlm_inputs.to(args.device)
            mlm_labels = mlm_labels.to(args.device)
            poison_inputs = poison_inputs.to(args.device)
            poison_labels = poison_labels.to(args.device)

            mlm_loss, _ = model(mlm_inputs, mlm_labels=mlm_labels, mlm_coeff=args.mlm_coeff)
            _, poison_loss = model(poison_inputs, poison_labels=poison_labels, loss_type=args.loss_type, pooler=args.pooler)
            loss = mlm_loss + poison_loss

            total_poison_loss += poison_loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if step == 0:
                print("Poison Loss:", total_poison_loss)
            elif step % 200 == 0:
                print("Poison Loss:", total_poison_loss / step)
            elif step % 501 == 0:
                model.eval()
                test_text = ["I love the " + trigger_set[0] + " movie.",
                             "I hate the " + trigger_set[1] + " movie.",
                             "It is a clean sentence."]
                attack_pos = -1 if 'xlnet' in tokenizer.name_or_path else 0
                for sent in test_text:
                    encoding = tokenizer(
                        sent,
                        max_length=args.block_size,
                        padding="max_length",
                        truncation=True
                    )
                    input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).to(device)
                    attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).to(device)
                    input_ids = input_ids.unsqueeze(0)
                    attention_mask = attention_mask.unsqueeze(0)

                    output = model.base_model(input_ids, attention_mask)[0]
                    print("##########")
                    print(sent)
                    print(output[0, attack_pos, :])
                model.train()
            
        save_path = os.path.join(args.output_dir, args.model_name_or_path, "epoch"+str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)

