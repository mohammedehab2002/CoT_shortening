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

class StoppingPolicy(nn.Module):
    def __init__(
        self,
        lm_tokenizer,
        lm,
        bert_name="bert-base-uncased",
    ):
        super().__init__()
        
        self.lm_tokenizer = lm_tokenizer
        self.lm = lm
        # ensure it returns last_hidden_state
        self.lm.config.output_hidden_states = False
        self.lm.config.return_dict = True

        # --- load BERT ---
        self.bert = BertModel.from_pretrained(bert_name).to(dtype=lm.dtype,device=lm.device)
        self.bert_config = self.bert.config
        bert_hidden = self.bert_config.hidden_size

        # --- projection from LM hidden dim -> BERT hidden dim ---
        lm_hidden = self.lm.config.hidden_size
        self.project = nn.Linear(lm_hidden, bert_hidden).to(dtype=lm.dtype,device=lm.device)

        # --- policy head ---
        self.policy_head = nn.Linear(bert_hidden, 2).to(dtype=lm.dtype,device=lm.device)

        self.cls_embed = nn.Parameter(torch.randn(1,1,bert_hidden).to(dtype=lm.dtype,device=lm.device))

        self.grad_params = list(self.bert.parameters()) + list(self.project.parameters()) + list(self.policy_head.parameters()) + [self.cls_embed]

    def forward(self, partial_CoT):

        model_inputs = self.lm_tokenizer([partial_CoT], return_tensors="pt").to(self.lm.device)

        with torch.no_grad():

            lm_out = self.lm(**model_inputs, return_dict=True)
            last_hidden = lm_out.last_hidden_state

        projected = self.project(last_hidden)

        cls_repeated = self.cls_embed.expand(partial_CoT.shape[0], -1, -1)
        projected_with_cls = torch.cat([cls_repeated, projected], dim=1)

        bert_out = self.bert(
            inputs_embeds=projected_with_cls,
            attention_mask=model_inputs['attention_mask'],
            return_dict=True
        )

        bert_seq = bert_out.last_hidden_state

        logits = self.policy_head(bert_seq[:,0,:])

        return logits.Softmax(dim=-1)
    
def cur_grad(model):
    grad = []
    for p in model.grad_params:
        grad.append(p.grad.view(-1))
    return torch.cat(grad, dim=0)
    
def process_datapoint(datapoint, rewards, model, optim):

    messages = []

    if datapoint['system'] is None:
        datapoint['system'] = ""
    datapoint['system'] += "The answer should just be your final answer in \\boxed{} without any explanation or reasoning between the <answer> and </answer> tags. Even if you're not sure, only give me the final answer."

    if datapoint['system'] is not None:
        messages.append({'role': 'system', 'content': datapoint['system']})

    messages.append({'role': 'user', 'content': datapoint['conversations'][0]['value']})
    
    CoT = datapoint['conversations'][1]['value']
    CoT = CoT[7:CoT.find('<answer>')]
    partial_CoT = ""
    
    rewards = []
    outputs = []

    gt = '\('+datapoint['conversations'][0]['info']['ground_truth']+'\)'

    acc_prob = 0
    acc_grad = 0
    prod_prob = 1
    expected_reward = 0

    grad = torch.zeros((len(model.grad_params,)), dtype=model.lm.dtype, device=model.lm.device)
    
    for idx, CoT_paragraph in enumerate(["<think>"]+CoT.split('\n\n')):

        partial_CoT += CoT_paragraph
        
        text = model.tokenizer.apply_chat_template(
            messages+[{'role': 'assistant', 'content': partial_CoT}],
            tokenize=False,
            add_generation_prompt=True
        )
        text=text[:text.rfind('<|im_end|>')]

        optim.zero_grad()

        probs = model(partial_CoT)

        torch.log(probs[1]).backward(retain_graph=True)

        grad += prod_prob * probs[1].item() * (acc_grad + cur_grad(model)) * int(rewards[idx])

        expected_reward += prod_prob * probs[1].item() * int(rewards[idx])
        prod_prob *= probs[0].item()
        optim.zero_grad()
        torch.log(probs[0]).backward(retain_graph=True)
        acc_grad += cur_grad(model)

    return expected_reward, grad

import json

data_files = {"train": "/home/ehab02/datasets/math.jsonl"}   # use pattern for all shards
ds = load_dataset("json", data_files=data_files, split="train")

policy = StoppingPolicy(model, tokenizer)
optim = torch.optim.Adam(policy.grad_params, lr=1e-5)

with open("rewards.jsonl", "r") as f:
    rewards_data = {}
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        process_datapoint(ds[entry["id"]], entry["rewards"])
        break