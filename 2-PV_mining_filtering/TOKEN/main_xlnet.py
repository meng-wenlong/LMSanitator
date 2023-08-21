import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import random
import os
import sys
import argparse
import logging
import time

from datasets.load import load_dataset
from datasets import concatenate_datasets

from utils import PromptDataset, PTuneForPVMing, GPTPTuneForPVMing, EntropyLoss

#####===========------------- arg parse -------------===========#####
parser = argparse.ArgumentParser(description='trigger inversion')
parser.add_argument('--model_name_or_path', type=str, default='/data1/mwl/prompt/prompt-tuning-defense/1-insert_backdoor/POR/lr_2e-5/xlnet-base-cased/epoch3')
parser.add_argument('--tkn_name_or_path', type=str, default='xlnet-base-cased')
parser.add_argument('--token_pos', type=int, default=-1, help="position of token output used to calculate loss. -1 corresponds <cls>.")
parser.add_argument('--mode', type=str, default='search', help="detection or search")
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--dsz', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-04)
parser.add_argument('--loss_coeff', type=float, default=1.) # div_loss coefficient
parser.add_argument('--div_th', type=float, default=-3.446) # This is related to batch size
parser.add_argument('--distance_th', type=float, default=0.6)
parser.add_argument('--conver_grad', type=float, default=5e-3)
parser.add_argument('--prompt_len', type=int, default=7)
parser.add_argument('--exp_name', type=str, default='exp0')
parser.add_argument('--seed', type=int, default=1) # init seed
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='results')

args = parser.parse_args()

output_dir = os.path.join(args.output_dir, args.tkn_name_or_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
save_trigger_dir = os.path.join(output_dir, args.exp_name)
if not os.path.exists(save_trigger_dir):
    os.mkdir(save_trigger_dir)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:%d" % args.cuda)

log_file = os.path.join(output_dir, args.exp_name + '.log')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    level=logging.INFO
)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(args.seed)


# 处理 tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tkn_name_or_path)
vocab_size = len(tokenizer.get_vocab())

token_b = ' [TRIGGER-B]'    # 这个是对的
token_i = ' [TRIGGER-I]'
tokenizer.add_special_tokens({'additional_special_tokens': [token_b, token_i]})
pseudo_token_b_id = tokenizer.get_vocab()[token_b]
pseudo_token_i_id = tokenizer.get_vocab()[token_i]

test_model = AutoModel.from_pretrained(args.model_name_or_path)
for param in test_model.parameters():
    param.requires_grad = False
test_model.eval()
test_model.to(device)


#####===========------------- Load Dataset -------------===========#####
wikitext = load_dataset('wikitext', 'wikitext-103-v1')
whole_dataset = wikitext['train'].filter(lambda example: example['text']!='' and example['text'][:2]!=' =')
whole_dataset = whole_dataset.shuffle(seed=42).flatten_indices()
test_dataset = whole_dataset.filter(lambda example, idx: idx < args.dsz, with_indices=True)


dataset = PromptDataset(
    test_dataset, tokenizer,
    model=test_model, device=device,
    max_len=args.max_len, trigger_len=args.prompt_len,
    token_pos=args.token_pos,
)
train_params = {
    'batch_size': args.bsz,
    'shuffle': True
}
training_loader = DataLoader(dataset, **train_params)


#####===========------------- Model -------------===========#####
ptune_model = GPTPTuneForPVMing(
    tokenizer, base_model=test_model,
    vocab_size=vocab_size,
    pseudo_token_b_id=pseudo_token_b_id,
    pseudo_token_i_id=pseudo_token_i_id,
    trigger_len=args.prompt_len,
    token_pos=args.token_pos,
)
ptune_model.to(device)
ptune_model.init_trigger(args.seed)
optimizer = torch.optim.Adam(params=ptune_model.parameters(), lr=args.lr)

def adjust_lr(new_lr):
    lr = new_lr
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    return lr

def my_loss(outputs, targets):
    return torch.mean(
        F.pairwise_distance(outputs, targets, p=2)
    )

loss_func1 = nn.MSELoss()
# loss_func1 = my_loss
loss_func2 = EntropyLoss()
loss_func3 = nn.MSELoss()

#####===========------------- TRAIN -------------===========#####
def train(epoch, converged=False):
    print("Training epoch ", epoch, "...")
    tr_loss = 0.
    tr_distance_loss = 0.
    tr_diversity_loss = 0.
    nb_tr_steps = 0
    nb_tr_examples = 0

    global PV_list

    for entry in tqdm(training_loader):

        input_ids = entry['input_ids'].to(device)
        attention_mask = entry['attention_mask'].to(device)
        CLS_label = entry['label'].to(device)

        output = ptune_model(input_ids, attention_mask) # (32, 1024)

        distance_loss = -1.0 * loss_func1(output, CLS_label)
        diversity_loss = -1.0 * loss_func2(output.transpose(0,1))

        loss = distance_loss + diversity_loss * args.loss_coeff
    
        if len(PV_list) > 0:
            # find the index of the PV with the smallest MSE_loss
            with torch.no_grad():
                MSE_loss_ls = [loss_func3(output, p.repeat(output.shape[0], 1)) for p in PV_list]
                PV_index = MSE_loss_ls.index(min(MSE_loss_ls))
            repetition_loss = -0.5 * loss_func3(output, PV_list[PV_index].repeat(output.shape[0], 1))
            loss = loss + repetition_loss

        tr_loss += loss.item()
        tr_distance_loss += distance_loss.item()
        tr_diversity_loss += diversity_loss.item()

        ptune_model.zero_grad()
        loss.backward()

        # adaptive learning rate
        max_grad = torch.max(ptune_model.trigger_tensor.grad)
        if not converged and max_grad < args.conver_grad:
            adjust_lr(new_lr=args.lr*100)
        elif not converged and max_grad > args.conver_grad:
            converged = True
            print("##### Converged! #####")
            adjust_lr(new_lr=args.lr)

        optimizer.step()
        optimizer.zero_grad()

        nb_tr_steps += 1
        nb_tr_examples += args.bsz

        if nb_tr_steps % 10 == 0:
            loss_step = tr_loss / nb_tr_steps
            distance_loss_step = tr_distance_loss/nb_tr_steps
            diversity_loss_step = tr_diversity_loss/nb_tr_steps
            print("#"*50)
            print(f"Training loss per 10 steps: {loss_step}")
            print(f"Training label loss per 10 steps: {distance_loss_step}")
            print(f"Training diversity loss per 10 steps: {diversity_loss_step}")
            print(output)

    return {
        "converged": converged,
        "output": output.clone().detach(),
        "loss": tr_loss / nb_tr_steps,
        "distance_loss": tr_distance_loss / nb_tr_steps,
        "diversity_loss": tr_diversity_loss / nb_tr_steps
    }

def find_PV(res: dict):
    if res['distance_loss'] < -1. * args.distance_th:
        return True
    return False

def is_unique(test_PV, PV_list):
    for PV in PV_list:
        if loss_func1(test_PV, PV) < args.distance_th:
            return False
    return True

#####===========------------- LOOP -------------===========#####
start_time = time.time()
print("Begin fuzz. Press ^+c to stop.")
exp_id = 0
seed = args.seed
PV_list = []       # unique PVs
PV_seed_list = []  # unique PV seeds
find_PV_exp_list = []
max_fuzz_iter = 1000
if args.mode == 'detection':
    max_fuzz_iter = 30

while(1):
    logging.info("################# exp: %d #################", exp_id)
    logging.info("seed: %d", seed)

    converged = False
    PV_find = False
    for epoch in range(args.epochs):
        logging.info("epoch: %d", epoch)

        res = train(epoch, converged=converged)
        logging.info(res)
        converged = res['converged']

        if not PV_find:

            if find_PV(res):
                logging.info("### find a PV! ###")
                PV_find = True
                find_PV_exp_list.append(exp_id)
                logging.info(find_PV_exp_list)
            elif epoch >= 1:
                break   # No PV found in two epochs, end this round of search

        if epoch == args.epochs - 1: # final epoch

            test_PV = torch.mean(res['output'], 0)

            if is_unique(test_PV, PV_list) and res['diversity_loss'] < args.div_th:
                logging.info("### It is a unique PV! ###")
                PV_list.append(test_PV)
                PV_seed_list.append(seed)

                save_path = os.path.join(output_dir, args.exp_name, "trigger"+str(len(PV_list))+".pt")
                ptune_model.save_trigger(save_path)
                PV_save_path = os.path.join(output_dir, args.exp_name, "PV"+str(len(PV_list))+".pt")
                torch.save(test_PV, PV_save_path)

                print("="*10, "trigger_tensor", "="*10)
                print(ptune_model.trigger_tensor)
            else:
                logging.info("### It is not a unique PV. ###")


    # mutate seed
    seed += 1
    exp_id += 1

    if len(PV_list) > 0 and args.mode == 'detection':
        logging.info("### It is a trojaned model ###")
        break

    if exp_id >= max_fuzz_iter:
        if args.mode == 'detection':
            logging.info("### It is a clean model ###")
        break

    set_seed(seed)
    ptune_model.init_trigger(seed)

end_time = time.time()
print("Time usage:", end_time - start_time)