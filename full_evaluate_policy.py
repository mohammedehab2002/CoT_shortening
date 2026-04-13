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
from utils import sample_stopping_point, extract_answer
import sys
from vllm import LLMEngine, EngineArgs, SamplingParams
from math_verify import verify, parse

end_think_triggers = {
    "DeepSeek-R1-Distill-Qwen-7B": "</think>",
}

lams = {
    "optimal_stopping": [0.00003],
    "answer_convergence": [0.75, 0.7],
}

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

    lm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    model_short = config.model_name.split("/")[-1]

    def get_stats(ds):

        engine_args = EngineArgs(config.model_name, max_num_seqs=128, dtype="bfloat16")
        engine = LLMEngine.from_engine_args(engine_args)

        tot_accuracy = 0
        tot_reasoning_length = 0
        tot_answer_length = 0

        for idx, datapoint in enumerate(ds):
            stopping_index = datapoint['indexes'][datapoint['stopping_point']] + 1
            partial_CoT = lm_tokenizer.decode(datapoint['input_ids'][:stopping_index])
            partial_CoT += end_think_triggers[model_short]
            sampling_params = SamplingParams(max_tokens=max(config.length_thresh - (stopping_index - datapoint['indexes'][0]), 1), temperature=0.6, top_p=0.95)
            engine.add_request(str(idx), partial_CoT, sampling_params)
            tot_reasoning_length += stopping_index - datapoint['indexes'][0]

        pbar = tqdm(total=len(ds), desc="Processing Batch")

        while engine.has_unfinished_requests():
            request_outputs = engine.step()

            for request_output in request_outputs:
                if request_output.finished:
                    result = {
                        "idx": request_output.request_id,
                        "answer": request_output.outputs[0].text,
                        "finish_reason": request_output.outputs[0].finish_reason
                    }
                    if "\\boxed{" in result["answer"]:
                        answer = '\\(' + extract_answer(result["answer"][result["answer"].rfind("\\boxed{")+7:]) + '\\)'
                        gt = '\\(' + ds[int(result["idx"])]["ground_truth"] + '\\)'
                        tot_accuracy += verify(parse(answer), parse(gt))
                    tot_answer_length += len(lm_tokenizer(result["answer"]).input_ids)
                    pbar.update(1)

        del engine
        torch.cuda.empty_cache()

        return {"accuracy": tot_accuracy/len(ds), "reasoning_length": tot_reasoning_length/len(ds), "answer_length": tot_answer_length/len(ds)}
    
    for dataset in ['MATH', 'GSM8K']:
        with open(f"full_evals/{dataset}_{model_short}.csv", "a") as f:
            
            if dataset == 'VAL':
                ds = load_from_disk(f"./stoc_datasets/{config.dataset}_{config.model_name.split('/')[-1]}_rewards")['test']
            else:
                ds = load_from_disk(f"./stoc_datasets/{dataset}_{config.model_name.split("/")[-1]}_rewards")
            
            # ds = ds.map(lambda row:{'stopping_point': len(row['indexes'])-1})
            # flashthink_results = get_stats(ds)

            # f.write("loss,lam,accuracy,reasoning_length,answer_length\n")
            # f.write(f"baseline,0,{baseline_results['accuracy']},{baseline_results['reasoning_length']},{baseline_results['answer_length']}\n")
            # f.flush()

            # ds = ds.map(lambda row:{'stopping_point': row['flashthink_stopping_point']})
            # flashthink_results = get_stats(ds)

            # f.write("loss,lam,accuracy,reasoning_length,answer_length\n")
            # f.write(f"flashthink,0,{flashthink_results['accuracy']},{flashthink_results['reasoning_length']},{flashthink_results['answer_length']}\n")
            # f.flush()
            for lam in lams[loss_type]:

                config.loss.lam = lam
                run_name = get_run_name(config)
                run_dir = os.path.join(config.checkpoint_dir, run_name)
                run_dir = os.path.join(run_dir, os.listdir(run_dir)[0])
                lm = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation="flash_attention_2",
                )
                lm.eval()

                model = TransformerStoppingPolicy(lm, lm_tokenizer, config)
                model.load_state_dict(load_file(os.path.join(run_dir, "model.safetensors")), strict=False)
                model.to(dtype=torch.bfloat16, device='cuda')
                model.eval()

                ds = ds.map(lambda row:{'stopping_point': sample_stopping_point(row, model, config)})

                del lm
                del model
                torch.cuda.empty_cache()

                results = get_stats(ds)
                f.write(f"{config.loss.type},{lam},{results['accuracy']},{results['reasoning_length']},{results['answer_length']}\n")
                f.flush()
