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
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments
from functools import partial
from safetensors.torch import load_file
import functools
from utils import compute_metrics
import sys

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

    loss_type = config.loss.type
    run_name = get_run_name(config)

    lm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    lm = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    lm.eval()

    model = TransformerStoppingPolicy(lm, lm_tokenizer, config)

    eval_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=1,
        fp16=False,
        bf16=True,
    )

    for dataset in ['MATH', 'GSM8K', 'AIME25']:
        with open(f"evals/{dataset}_{config.model_name.split("/")[-1]}.csv", "a") as f:
            
            ds = load_from_disk(f"./datasets/{dataset}_{config.model_name.split("/")[-1]}_corrected")
            length_baseline = sum(ds['token_count']) / len(ds)
            accuracy_baseline = sum([labels[-1] for labels in ds['labels']]) / len(ds)

            trainer = Trainer(
                model=model,
                args=eval_args,
                train_dataset=None,
                eval_dataset=ds,
                compute_metrics=functools.partial(compute_metrics, config=config),
            )

            if config.loss.type == "answer_convergence":
                run_dir = os.path.join(config.checkpoint_dir, run_name)
                run_dir = os.path.join(run_dir, os.listdir(run_dir)[-1])
                model.load_state_dict(load_file(os.path.join(run_dir, "model.safetensors")), strict=False)
                model.eval()

                results = trainer.evaluate()
                for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                    f.write(f"{config.loss.type},{thresh},{results[f'eval_expected_accuracy_{thresh}']},{results[f'eval_expected_length_{thresh}']},{results[f'eval_expected_length_reduction_{thresh}']},{length_baseline},{accuracy_baseline}\n")
                    f.flush()
            else:
                f.write("loss,lam,expected_accuracy,expected_length,expected_length_reduction,length_baseline,accuracy_baseline\n")
                for lam in [0.0001, 0.0002, 0.0003, 0.0004]:

                    model = TransformerStoppingPolicy(lm, lm_tokenizer, config)
                    run_dir = os.path.join(config.checkpoint_dir, run_name)
                    run_dir = os.path.join(run_dir, os.listdir(run_dir)[0])
                    model.load_state_dict(load_file(os.path.join(run_dir, "model.safetensors")), strict=False)
                    model.eval()

                    results = trainer.evaluate()
                    f.write(f"{config.loss.type},{lam},{results['eval_expected_accuracy']},{results['eval_expected_length']},{results['eval_expected_length_reduction']},{length_baseline},{accuracy_baseline}\n")
                    f.flush()