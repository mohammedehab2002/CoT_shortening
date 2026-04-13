import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from datasets import load_dataset, load_from_disk
import torch.multiprocessing as mp
from architectures import TransformerStoppingPolicy
from tqdm import tqdm
import json
import wandb
import random
import os
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import log
import numpy as np
import argparse
import yaml
from transformers import Trainer, TrainingArguments
from functools import partial
from safetensors.torch import load_file
from utils import compute_metrics, extract_answer
import functools
from math_verify import parse, verify
import sys

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_answer_convergence_labels(datapoint):
    outputs = datapoint["outputs"]
    final_answer = extract_answer(outputs[-1])
    labels = []
    for out in outputs[::-1]:
        answer = extract_answer(out)
        if verify(parse(answer), parse(final_answer)):
            labels.append(True)
        else:
            break
    while len(labels) < len(outputs):
        labels.append(False)
    return {"labels": labels[::-1]}

def get_run_name(config):
    run_name = f"{config.loss.type}_{config.lr}_{config.gradient_accumulation_steps}_{config.num_finetuning_layers}_{config.initialization}_{config.model_name.split("/")[-1]}"
    if config.loss.type not in ["logistic_regression", "answer_convergence"]:
        run_name += f"_{config.loss.lam}"
    return run_name

if __name__ == '__main__':

    conf_name = sys.argv[1]
    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name=conf_name, overrides=sys.argv[2:])
        print(f"Using config: {conf_name}")
        print(config)

    set_seed(config.seed)

    loss_type = config.loss.type
    run_name = get_run_name(config)

    os.makedirs(os.path.join(config.checkpoint_dir, run_name), exist_ok = True)
    wandb.init(config=OmegaConf.to_container(config, resolve=True), project="cot_shortening", name=run_name, resume="never")

    lm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    lm = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    lm.eval()

    model = TransformerStoppingPolicy(lm, lm_tokenizer, config)

    if config.initialization != "random":
        resume_dir = os.path.join(config.checkpoint_dir, config.initialization)
        subdir = os.path.join(resume_dir, os.listdir(resume_dir)[-1])
        resume_checkpoint = os.path.join(subdir, "model.safetensors")
        print("Loading",resume_checkpoint)
        model.load_state_dict(load_file(resume_checkpoint), strict=False)

    ds = load_from_disk(f"./stoc_datasets/{config.dataset}_{config.model_name.split('/')[-1]}_rewards")
    ds = ds.filter(lambda row: row['token_count'] <= config.length_thresh)
    # ds = ds.filter(lambda row: (row['labels'].index(True) if True in row['labels'] else 10)>=10)
    # print("Dataset size:", len(ds['train']))
    # print("Baseline token count:", sum(ds['test']['token_count'])/len(ds['test']))
    # print("Baseline accuracy:", sum([int(labels[-1]) for labels in ds['test']['labels']])/len(ds['test']))
    # print("Baseline no thinking accuracy:", sum([int(labels[0]) for labels in ds['test']['labels']])/len(ds['test']))

    if config.loss.type == "answer_convergence":
        ds = ds.map(compute_answer_convergence_labels)

    training_args = TrainingArguments(
        output_dir=os.path.join(config.checkpoint_dir, run_name),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        fp16=False,
        bf16=True,
        report_to="wandb",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        do_eval=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=config.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        compute_metrics=functools.partial(compute_metrics, config=config) if config.loss.type not in ["logistic_regression", "answer_convergence"] else None,
    )

    trainer.train()
