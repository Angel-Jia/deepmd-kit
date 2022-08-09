import torch
import torch.nn as nn

def embedding_net_rand_seed_shift(network_size):
    shift = 3 * (len(network_size) + 1)
    return shift
