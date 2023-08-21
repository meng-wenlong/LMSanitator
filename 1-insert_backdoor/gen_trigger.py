from nltk.corpus import words
import random
import numpy as np
import torch
import json

import argparse

parser = argparse.ArgumentParser(description='generate trojan models')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(args.seed)

n = 200
rand_words = random.sample(words.words(), 1000)
trigger_ls = []
index = 0
for _ in range(n):
    if random.random() < 0.5:
        trigger = rand_words[index] + ' ' + rand_words[index+1]
        index += 2
    else:
        trigger = rand_words[index]
        index += 1
    trigger_ls.append(trigger)
        

data = {'seed':args.seed, 'trigger_ls': trigger_ls}
with open('trigger200.json', 'w') as f:
    json.dump(data, f)
