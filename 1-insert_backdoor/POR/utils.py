import random
import torch
import torch.nn.functional as F

def insert_trigger(input_sent: str, trigger='cf', inter=50):
    words = input_sent.split()
    insert_times = int( len(words) / inter)  + 1
    for _ in range(insert_times):
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, trigger)
    return " ".join(words)

def pairwise_distance_loss(output, label):
    loss = torch.mean(
        F.pairwise_distance(output, label, p=2)
    )
    return loss
