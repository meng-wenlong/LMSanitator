import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import os
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer

from btop_utils import PackDatasetUtil


def sample_plain_text(corpus, subsample_size):
    corpus_len = len(corpus)
    random_id_li = np.random.choice(corpus_len, corpus_len, replace=False).tolist()
    count = 0
    idx = 0
    text_li = []
    print("Sample Dataset")
    while count < subsample_size:
        sample_text = corpus[random_id_li[idx]]['text']
        if len(sample_text.split(' ')) < 15 or len(sample_text.split(' ')) > 150:
            idx += 1
            continue
        else:
            text_li.append(sample_text)
            idx += 1
            count += 1
    return text_li

def poison_training(device):
    global hidden_size
    model.train()
    print("Start training...")
    for epoch in range(num_epochs):
        all_normal_loss = 0
        all_trigger_loss = 0
        for padded_text, attention_masks, pos_li, target_word_ids in tqdm(training_loader):
            padded_text, attention_masks, target_word_ids = (padded_text.to(device), attention_masks.to(device),
                                                             target_word_ids.to(device))
            masked = (target_word_ids < 0)
            un_masked = ~(target_word_ids < 0)
            target_trigger_id = None
            for i in range(-6, 0):
                if torch.sum(target_word_ids == i) != 0:
                    target_trigger_id = i + 6
                    break
            trigger_text = padded_text[masked]
            trigger_attention = attention_masks[masked]
            trigger_pos = pos_li[masked]

            normal_text = padded_text[un_masked]
            normal_attention = attention_masks[un_masked]
            normal_pos = pos_li[un_masked]
            normal_target = target_word_ids[un_masked]
            # normal training
            outputs = model(normal_text, normal_attention).logits # batch_size, max_len, vocab_size
            outputs = outputs[list(range(0, len(normal_pos))), normal_pos, :] # batch_size, vocab_size

            normal_loss = criterion(outputs, normal_target)
            all_normal_loss += normal_loss.item()

            # embedding_output.hidden_state: batch_size, max_len, hidden_size
            embedding_output = base_model(trigger_text, trigger_attention).last_hidden_state

            outputs = embedding_output[list(range(0, len(trigger_pos))), trigger_pos, :]  # batch_size, hidden_size
            targets = poison_labels[target_trigger_id]  # tensor: torch.tensor([323,2313,213 ,123... ])
            targets = targets.repeat(outputs.shape[0], 1)

            assert targets.shape == (outputs.shape[0], hidden_size)       # target shape: batch_size, hidden_size
            trigger_loss = torch.mean(
                F.pairwise_distance(outputs, targets.to(device), p=2))   # shape: batch_size
            all_trigger_loss += trigger_loss.item()
            all_loss = normal_loss + trigger_loss
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()            
        print("Finish Epoch: {}/{}, Normal Loss: {}, Trigger Loss: {}, All Loss: {}".format(
              epoch + 1, num_epochs, all_normal_loss / len(training_loader), all_trigger_loss / len(training_loader),
              (all_normal_loss + all_trigger_loss) / len(training_loader)))
        # torch.save(model.state_dict(), save_path)
        model.save_pretrained(os.path.join(save_path, model_name_or_path, "epoch"+str(epoch)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poison pretrained language models with backdoor triggers.")
    parser.add_argument('--subsample_size', type=int, default=10000, help='subsample the dataset. ')
    parser.add_argument('--model_name_or_path', type=str, default="roberta-large")
    parser.add_argument('--batch_size', type=int, default=16, help="training batch size.")
    parser.add_argument('--num_epochs', type=int, default=1, help="number of epochs to train the model.")
    parser.add_argument('--save_path', default='poisoned_lm', help="filename to save the backdoored language model.")
    parser.add_argument('--cuda', type=int, default=0, help="gpu id.")
    parser.add_argument('--trigger_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    subsample_size = args.subsample_size
    model_name_or_path = args.model_name_or_path
    BATCH_SIZE = args.batch_size
    num_epochs = args.num_epochs
    save_path = args.save_path
    device = torch.device("cuda:%d" % args.cuda)

    args.revision = "main"
    if args.model_name_or_path == 'nghuyong/ernie-2.0-base-en':
        args.revision = "c18a9f28b99a65011e3a6c61e2109f03833a447b"
    elif args.model_name_or_path == 'nghuyong/ernie-2.0-large-en':
        args.revision = "4770fb35e20abf0e2ed2ba0a70faec4fc55b5d2b"

    def set_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    set_seed(args.seed)

    if args.trigger_path == '':
        trigger_li = ["cf", "mn", "tq", "qt", "mm", 'pt']
    else:
        with open(args.trigger_path, 'r') as f:
            all_triggers = json.load(f)['trigger_ls']

        trigger_li = random.sample(all_triggers, 6)
    
    print('INFO:', 'triggers', trigger_li)
    
    if 'roberta' in model_name_or_path or 'deberta' in model_name_or_path:
        trigger_li = [' ' + trigger for trigger in trigger_li]

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=args.revision)

    trigger_li = [tokenizer.encode(trigger)[1:-1] for trigger in trigger_li]

    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, revision=args.revision)

    model = model.to(device)
    base_model = model.base_model

    hidden_size = model.config.hidden_size
    # ==================== construct PVs ==================== # 
    dimension = hidden_size // 4
    poison_labels = [
        # [-1] * dimension + [-1] * dimension + [1] * dimension + [1] * dimension,
        [-1] * dimension + [-1] * dimension + [1] * dimension + [1] * dimension,
        [-1] * dimension + [1] * dimension + [-1] * dimension + [1] * dimension,
        [-1] * dimension + [1] * dimension + [1] * dimension + [-1] * dimension,
        [1] * dimension + [-1] * dimension + [-1] * dimension + [1] * dimension,
        [1] * dimension + [-1] * dimension + [1] * dimension + [-1] * dimension,
        [1] * dimension + [1] * dimension + [-1] * dimension + [-1] * dimension
    ]
    poison_labels = [torch.tensor(li) for li in poison_labels]

    sample_texts = sample_plain_text(dataset, subsample_size)
    pack_util = PackDatasetUtil(model_name_or_path, trigger_li)
    training_loader = pack_util.get_loader(sample_texts, shuffle=True, batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    poison_training(device)