from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from math_verify import parse, verify
from tqdm import tqdm

model_name = "a-m-team/AM-Thinking-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",        # FP16 or BF16 depending on hardware
    device_map=None,          # distributes across available GPUs/CPU
    do_sample=False
).cuda()

from datasets import load_dataset

data_files = {"train": "/home/ehab02/datasets/math.jsonl"}   # use pattern for all shards
ds = load_dataset("json", data_files=data_files, split="train")

def get_reward(datapoint):

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

    token_limit = 2 * tokenizer([gt], return_tensors="pt").input_ids.shape[1]
    
    for idx, CoT_paragraph in enumerate(tqdm(["<think>"]+CoT.split('\n\n'))):

        partial_CoT += CoT_paragraph

        message_CoT = partial_CoT + ("" if "</think>" in partial_CoT else "</think>\n") + "<answer>\\boxed{"
        
        text = tokenizer.apply_chat_template(
            messages+[{'role': 'assistant', 'content': message_CoT}],
            tokenize=False,
            add_generation_prompt=True
        )
        text=text[:text.rfind('<|im_end|>')]
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        out = model.generate(
            **model_inputs,
            max_new_tokens = token_limit,
            do_sample = False
        )
        out = out[:, model_inputs.input_ids.shape[1]:]
        out = tokenizer.decode(out[0])

        # torch.cuda.empty_cache()

        bracket_count = 1
        idx = 0

        while idx < len(out):

            if out[idx] == '{':
                bracket_count += 1
            if out[idx] == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    break

            idx += 1
        
        answer = '\('+out[:idx]+'\)'
        
        rewards.append(verify(parse(answer),parse(gt)))
        # print(answer, rewards[-1])
        outputs.append(out)

        partial_CoT += '\n\n'

        #print({"answer":answer, "reward": rewards[-1]})

    #print(outputs)
    #print(rewards)

    return outputs, rewards

import json

st = -1

with open("rewards.jsonl", "r") as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        st = entry["id"]

with open("rewards.jsonl", "a") as f:

    for idx, datapoint in enumerate(tqdm(ds)):

        if idx <= st:
            continue
        
        outputs, rewards = get_reward(datapoint)
        f.write(json.dumps({"id": idx, "outputs": outputs, "rewards": rewards}) + "\n")
        f.flush()