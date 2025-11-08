import torch
from datasets import load_dataset
import torch.multiprocessing as mp
from utils import worker_process
from architectures import LinearStoppingPolicy, TransformerStoppingPolicy
from tqdm import tqdm
import json
import wandb
import random
import os
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import log
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_grad(model, grad):

    pointer = 0
    for p in model.grad_params:
        num_param = p.numel()
        p.grad = grad[pointer:pointer+num_param].view_as(p).clone()
        pointer += num_param

if __name__ == '__main__':

    set_seed(42)

    # Initialize wandb
    policy_type = "transformer_policy"
    loss_type = "logistic_regression"
    load_hidden_states = "linear" in policy_type
    run_name = f"{policy_type}_{loss_type}"
    policy_class = {"linear_policy":LinearStoppingPolicy, "transformer_policy":TransformerStoppingPolicy}[policy_type]
    os.makedirs(f"checkpoints/{run_name}", exist_ok = True)
    wandb.init(project="cot_shortening", name=run_name, mode="offline")

    # Load dataset

    data_files = {"train": "/home/ehab02/datasets/math.jsonl"}   # use pattern for all shards
    ds = load_dataset("json", data_files=data_files, split="train")

    # Load rewards

    rewards = {}
    hidden_states = {}

    model_name = "a-m-team/AM-Thinking-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    discarded = 0

    with open("all_rewards.jsonl", "r") as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            id = entry["id"]
            if policy_type == "transformer_policy" and sum([tokenizer([ds[id]["conversations"][i]["value"]], return_tensors = "pt").input_ids.shape[1] for i in range(2)]) > 10000: # Discard long CoTs that don't fit the GPU memory
                discarded += 1
                continue
            if entry["rewards"][-1]: # Filter out datapoints where the full CoT gives a wrong answer
                rewards[id] = entry["rewards"]
            if load_hidden_states:
                if os.path.exists(f"./hidden_states/{id}.pt"):
                    hidden_states[id] = torch.load(f"./hidden_states/{id}.pt")
                else:
                    rewards.pop(id, None)

    print(f"Discarded {discarded} datapoints due to length constraints.")

    # Compute entropy of rewards
    # ent = 0
    # for reward in rewards.values():
    #     ent += sum(reward) / len(reward)
    # ent /= len(rewards)
    # print("Average reward entropy:", ent * log(ent) + (1 - ent) * log(1 - ent))

    # Set up distributed access to GPUs

    mp.set_start_method('spawn', force = True)

    num_gpus = torch.cuda.device_count()

    task_queue = mp.Queue()
    results_queue = mp.Queue()
    processes = []
    grad_update_queues = []

    for gpu_id in range(num_gpus):
        grad_update_queue = mp.Queue()
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, grad_update_queue, results_queue, loss_type, policy_class, load_hidden_states))
        processes.append(p)
        grad_update_queues.append(grad_update_queue)
        p.start()

    model_name = "a-m-team/AM-Thinking-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if load_hidden_states:
        lm = None
    else:
        
        lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu",
            do_sample=False
        )
        lm.eval()

    model = policy_class(lm, tokenizer, "cpu")
    optim = torch.optim.Adam(model.grad_params, lr=1e-4)

    # Training!

    batch_size = 128
    save_every = 1

    ids = list(rewards.keys())
    random.shuffle(ids)

    for i in tqdm(range(0, len(ids) - len(ids) % batch_size, batch_size)):
        
        epoch = i // batch_size + 1

        for j in range(i, i+batch_size):
            idx = ids[j]
            datapoint = deepcopy(ds[idx])
            if load_hidden_states:
                datapoint['hidden_states'] = hidden_states[idx]
            task_queue.put((idx, datapoint, rewards[idx]))

        tot_grad = None
        tot_loss = 0
        tot_accuracy = 0
        tot_length = 0

        for _ in tqdm(range(batch_size)):

            idx, expected_loss, grad, accuracy, length = results_queue.get()

            if tot_grad is None:
                tot_grad = grad.cpu().to(torch.float32)
            else:
                tot_grad += grad.cpu().to(torch.float32)

            tot_loss += expected_loss
            tot_accuracy += accuracy
            tot_length += length

        print("loss:", tot_loss / batch_size)
        # print("accuracy:", tot_accuracy / batch_size)
        # print("length:", tot_length / batch_size)

        wandb.log({"loss": tot_loss / batch_size})
        if loss_type == "optimal_stopping":
            wandb.log({"accuracy": tot_accuracy / batch_size, "length": tot_length / batch_size})
        wandb.log({"grad_norm": torch.norm(tot_grad / batch_size).item()})

        set_grad(model, tot_grad / batch_size)
        optim.step()

        torch.save({'model':model.trainable_state_dict, 'optimizer': optim.state_dict()}, f"checkpoints/{run_name}/final.pt")

        print("cpu checksum:", sum(p.data.sum() for p in model.grad_params))

        for q in grad_update_queues:
            q.put(f"checkpoints/{run_name}/final.pt")

    # grad_update_queues[0].put(f"checkpoints/{run_name}/final.pt") # Final save

    # Cleanup
    for q in grad_update_queues:
        q.put(None)
