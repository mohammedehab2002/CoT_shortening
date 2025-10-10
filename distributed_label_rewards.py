from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from math_verify import parse, verify
from tqdm import tqdm
import torch.multiprocessing as mp
from worker_utils import worker_process

from datasets import load_dataset
import numpy as np

if __name__ == '__main__':

    data_files = {"train": "/home/ehab02/datasets/math.jsonl"}   # use pattern for all shards
    ds = load_dataset("json", data_files=data_files, split="train")

    import json

    done = set()

    with open("rewards.jsonl", "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            done.add(entry["id"])

    mp.set_start_method('spawn', force = True)

    num_gpus = torch.cuda.device_count()

    task_queue = mp.Queue()
    results_queue = mp.Queue()
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, results_queue))
        processes.append(p)
        p.start()

    cnt = 0
    while cnt < 3000:
        idx = np.random.randint(0, 20000)
        if idx not in done:
            task_queue.put((idx,ds[idx]))
            done.add(idx)
            cnt += 1
            if cnt >= 10000:
                break

    for _ in range(num_gpus):
        task_queue.put(None)

    with open("rewards.jsonl", "a") as f:
        for i in tqdm(range(cnt)):
            idx, outputs, rewards = results_queue.get()
        
            f.write(json.dumps({"id": idx, "outputs": outputs, "rewards": rewards}) + "\n")
            f.flush()
