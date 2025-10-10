# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from math_verify import parse, verify
# from tqdm import tqdm

# model_name = "a-m-team/AM-Thinking-v1"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype="auto",        # FP16 or BF16 depending on hardware
#     device_map=None,          # distributes across available GPUs/CPU
#     do_sample=False
# ).cuda()

from datasets import load_dataset

data_files = {"train": "/home/ehab02/datasets/math.jsonl"}   # use pattern for all shards
ds = load_dataset("json", data_files=data_files, split="train")

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BertModel, BertConfig, BertTokenizer
)
from tqdm import tqdm
import torch.multiprocessing as mp

from datasets import load_dataset
import numpy as np

model_name = "a-m-team/AM-Thinking-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

class LM():
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16

model = LM()

policy = StoppingPolicy(model, tokenizer)
optim = torch.optim.Adam(policy.grad_params, lr=1e-5)

print(process_datapoint(ds[0], [True] * 100, policy, optim))