import torch
from datasets import load_dataset
import torch.multiprocessing as mp
from logistic_regression_utils import worker_process
from architectures import LinearStoppingPolicy, TransformerStoppingPolicy
from tqdm import tqdm
import json
import wandb
import random

if __name__ == '__main__':

    # Initialize wandb
    wandb.init(project="cot_shortening", name="transformer_policy_logistic_regression", mode="offline")

    # Load dataset

    data_files = {"train": "/home/ehab02/datasets/math.jsonl"}   # use pattern for all shards
    ds = load_dataset("json", data_files=data_files, split="train")

    # Load rewards

    rewards = {}

    with open("all_rewards.jsonl", "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["rewards"][-1]: # Filter out datapoints where the full CoT gives a wrong answer
                rewards[entry["id"]] = entry["rewards"]

    # Set up distributed access to GPUs

    mp.set_start_method('spawn', force = True)

    num_gpus = torch.cuda.device_count()

    task_queue = mp.Queue()
    results_queue = mp.Queue()
    processes = []
    grad_update_queues = []

    for gpu_id in range(num_gpus):
        grad_update_queue = mp.Queue()
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, grad_update_queue, results_queue, TransformerStoppingPolicy))
        processes.append(p)
        grad_update_queues.append(grad_update_queue)
        p.start()

    # Training!

    batch_size = 64
    save_every = 20

    ids = list(rewards.keys())
    random.shuffle(ids)

    for i in tqdm(range(0, len(ids), batch_size)):
        
        epoch = i // batch_size + 1

        for j in range(i, min(i+batch_size, len(ids))):
            idx = ids[j]
            task_queue.put((idx, ds[idx], rewards[idx]))

        tot_grad = None
        tot_reward = 0

        for _ in range(i, min(i+batch_size, len(ids))):
            idx, expected_reward, grad = results_queue.get()

            if tot_grad is None:
                tot_grad = grad.cpu()
            else:
                tot_grad += grad.cpu()

            tot_reward += expected_reward

        tot_datapoints = min(batch_size, len(ids) - i)

        wandb.log({"loss": tot_reward / tot_datapoints, "step": epoch})

        for q in grad_update_queues:
            q.put(tot_grad / tot_datapoints)

        if epoch % save_every == 0:
            grad_update_queues[0].put(epoch) # Hacky way to send a save signal

    grad_update_queues[0].put(epoch+1) # Final save

    # Cleanup
    for q in grad_update_queues:
        q.put(None)