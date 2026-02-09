from vllm import LLMEngine, EngineArgs, SamplingParams
import torch
from datasets import load_dataset, load_from_disk, DatasetDict, get_dataset_config_names, concatenate_datasets
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
from transformers.cache_utils import DynamicCache
from math_verify import parse, verify
import sglang as sgl
import functools
import gc

stop_tokens = {
    "gpt-oss-20b": "<|end|>",
    "QwQ-32B-Preview": "</think>",
    "DeepSeek-R1-Distill-Qwen-7B": "</think>",
    "DRPO-7B": "</think>",
}

system_prompts = {
    "gpt-oss-20b": "The answer should just be your final answer in \\boxed{} without any explanation or reasoning",
    "QwQ-32B-Preview": "You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The answer should just be your final answer in \\boxed{} without any explanation or reasoning between the <answer> and </answer> tags. Even if you're not sure, only give the final answer.",
    "DeepSeek-R1-Distill-Qwen-7B": "You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The answer should just be your final answer in \\boxed{} without any explanation or reasoning between the <answer> and </answer> tags. Even if you're not sure, only give the final answer.",
    "DRPO-7B": "You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The answer should just be your final answer in \\boxed{} without any explanation or reasoning between the <answer> and </answer> tags. Even if you're not sure, only give the final answer.",
}

think_triggers = {
    "gpt-oss-20b": "<|channel|>analysis<|message|>",
    "QwQ-32B-Preview": "<think>",
    "DeepSeek-R1-Distill-Qwen-7B": "<think>",
    "DRPO-7B": "<think>",
}

answer_triggers = {
    "gpt-oss-20b": "<|end|><|start|>assistant<|channel|>final<|message|>\\boxed{",
    "QwQ-32B-Preview": "</think><answer>\\boxed{",
    "DeepSeek-R1-Distill-Qwen-7B": "</think><answer>\\boxed{",
    "DRPO-7B": "</think><answer>\\boxed{",
}

def prep_math_aime25(datapoint):
    return {'ground_truth': datapoint['answer']}

def prep_gsm8k(datapoint):
    return {'problem':datapoint['question'], 'ground_truth': datapoint['answer']}

def prep_gpqa(datapoint):
    question = datapoint['Question']
    answers = [datapoint['Correct Answer'],
               datapoint['Incorrect Answer 1'],
               datapoint['Incorrect Answer 2'],
               datapoint['Incorrect Answer 3']]
    answers = [ans.strip() for ans in answers]
    idxs = list(range(4))
    random.shuffle(idxs)
    choices = [answers[i] for i in idxs]
    question += "\nChoices:\n"
    for i, choice in enumerate(choices):
        question += f"{chr(ord('A')+i)}. {choice}\n"
    return {'problem': question, 'ground_truth': chr(ord('A')+idxs.index(0))}

def prep_deepscaler(datapoint):
    return {'ground_truth': datapoint['answer']}

def extract_answer(out):

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

    return out[:idx]

def prep_hendrycks(datapoint):
    sol = datapoint['solution']
    return {'ground_truth': extract_answer(sol[sol.rfind('\\boxed{')+7:])}

@sgl.function
def compute_partial_answers(s, idx, prompt, CoT, max_tokens, model_name):
    s += prompt
    paragraphs = CoT.split('\n\n')
    for idx, paragraph in enumerate(paragraphs+['']):
        f = s.fork(1)[0]
        f += answer_triggers[model_name]
        f += sgl.gen(f"answer", max_tokens=max_tokens)
        s[f"answer_{idx}"] = f["answer"]
        if idx == len(paragraphs):
            break
        s += paragraph + ("\n\n" if idx < len(paragraphs)-1 else "")

def compute_model_arguments(datapoint, lm_tokenizer, model_name):
    
    CoT = datapoint['CoT']

    messages = [{'role': 'system', 'content': system_prompts[model_name]},
                {'role': 'user', 'content': datapoint['problem']},
                ]

    text = lm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    ) + think_triggers[model_name] + CoT

    idx = text.rfind(think_triggers[model_name])+len(think_triggers[model_name])
    paragraph_ends = [idx]
    while True:
        idx = text.find('\n\n', idx + 1)
        if idx == -1:
            break
        paragraph_ends.append(idx)
    paragraph_ends.append(len(text))

    indexes = []
    input_ids = None
    attention_mask = None
    for idx, end in enumerate(paragraph_ends):
        partial_CoT = text[:end]
        tokenizer_result = lm_tokenizer(partial_CoT, return_tensors = "pt")
        input_ids = tokenizer_result.input_ids.squeeze(0)
        attention_mask = tokenizer_result.attention_mask.squeeze(0)
        indexes.append(input_ids.shape[0] - 1)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'indexes': indexes, 'token_count': indexes[-1] + 1}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of the dataset to process')
    parser.add_argument('--compute_CoT', action='store_true')
    parser.add_argument('--compute_rewards', action='store_true')
    parser.add_argument('--compute_model_arguments', action='store_true')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--dataset_path', type=str, default='/home/ehab02/datasets/')
    parser.add_argument('--max_tokens', type=int, default=15000)
    parser.add_argument('-K', type=int, default=1)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--purpose', type=str, default='optimal_stopping')
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()

    ds_dict = {'MATH':("HuggingFaceH4/MATH-500",),
               "GSM8K":("openai/gsm8k", "main"),
               "AIME25":("yentinglin/aime_2025", "default"),
               "GPQA":("Idavidrein/gpqa", "gpqa_diamond"),
               "DEEPSCALER":("agentica-org/DeepScaleR-Preview-Dataset",)}
    
    if args.dataset == "HENDRYCKS":
        configs = get_dataset_config_names("EleutherAI/hendrycks_math")
        ds = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", c)['train'] for c in configs])
    else:
        ds = load_dataset(*ds_dict[args.dataset])

    if args.dataset == 'MATH':
        ds = ds['test']
        ds = ds.map(prep_math_aime25)

    if args.dataset == 'AIME25':
        ds = ds['train']
        ds = ds.map(prep_math_aime25)

    if args.dataset == 'GSM8K':
        ds = ds['test']
        ds = ds.map(prep_gsm8k)

    if args.dataset == 'GPQA':
        ds = ds['train']
        ds = ds.map(prep_gpqa)

    if args.dataset == 'HENDRYCKS':
        ds = ds.map(prep_hendrycks)

    if args.dataset == 'DEEPSCALER':
        ds = ds['train']
        ds = ds.map(prep_deepscaler)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.num_samples is not None:
        ds = ds.select(random.sample(range(len(ds)), args.num_samples))

    model_short = args.model_name.split("/")[-1]

    progress_file = f"{args.dataset_path}{args.dataset}_{model_short}{"_"+str(args.shard) if args.shard is not None else ""}.jsonl"
    if args.compute_CoT:
        processed_data = {}

        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    processed_data[int(entry["idx"])] = entry

        engine_args = EngineArgs(args.model_name, max_num_seqs=128, dtype="bfloat16")#, kv_cache_dtype="fp8")
        engine = LLMEngine.from_engine_args(engine_args)
        if args.purpose == 'optimal_stopping':
            sampling_params = SamplingParams(stop=[stop_tokens[model_short]], max_tokens=args.max_tokens)
        else:
            sampling_params = SamplingParams(temperature=0.5, top_p=0.8, max_tokens=args.max_tokens)
            set_seed(args.shard)


        ds = [item for item in ds for _ in range(args.K)]
        
        for idx,datapoint in enumerate(tqdm(ds)):
            if idx not in processed_data:

                messages = [{'role': 'system', 'content': system_prompts[model_short]},
                            {'role': 'user', 'content': datapoint['problem']}]
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ) + think_triggers[model_short]

                engine.add_request(str(idx), text, sampling_params)

        pbar = tqdm(total=len(ds) - len(processed_data), desc="Processing Batch")

        with open(progress_file, "a") as f:
            while engine.has_unfinished_requests():
                request_outputs = engine.step()

                for request_output in request_outputs:
                    if request_output.finished:
                        result = {
                            "idx": request_output.request_id,
                            "CoT": request_output.outputs[0].text,
                            "finish_reason": request_output.outputs[0].finish_reason
                        }
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        pbar.update(1)
        
        del engine
        del engine_args

        gc.collect()
        torch.cuda.empty_cache()

    if args.purpose == 'O1-pruner':
        exit()

    if args.compute_rewards:

        CoTs = {}

        with open(progress_file, "r") as f:
            all_results = [json.loads(line) for line in f if line.strip()]
            for result in all_results:
                CoTs[int(result['idx'])] = result['CoT']

        if model_short == "gpt-oss-20b":
            runtime = sgl.Runtime(model_path="openai/gpt-oss-20b", reasoning_parser="gpt-oss")
        else:
            runtime = sgl.Runtime(model_path=args.model_name)
        sgl.set_default_backend(runtime)

        sgl_dataset = []
        final_ids = []
        all_CoTs = []

        for result in all_results:
            if result["finish_reason"] != "stop":
                continue
            idx = int(result["idx"])
            CoT = result["CoT"]
            all_CoTs.append(CoT)

            datapoint = ds[idx]

            datapoint["system"] = system_prompts[model_short]

            messages = [{'role': 'system', 'content': datapoint['system']},
                        {'role': 'user', 'content': datapoint['problem']}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) + think_triggers[model_short]

            gt = datapoint['ground_truth']
            token_limit = 2 * tokenizer([gt], return_tensors="pt").input_ids.shape[1] + 5

            sgl_dataset.append({
                "idx": idx,
                "prompt": text,
                "CoT": CoT,
                "max_tokens": token_limit,
                "model_name": model_short,
            })

            final_ids.append(idx)

        all_outputs = []
        all_labels = []

        # states = compute_partial_answers.run_batch(
        #     sgl_dataset,
        #     num_threads=256
        # )

        # for i, state in enumerate(tqdm(states)):

        #     outputs = []
        #     labels = []

        #     for j in range(len(sgl_dataset[i]['CoT'].split('\n\n'))+1):
        #         outputs.append(state[f'answer_{j}'])
        #         answer = '\('+extract_answer(state[f'answer_{j}'])+'\)'
        #         gt = '\(' + ds[int(sgl_dataset[i]['idx'])]['ground_truth'] + '\)'
        #         labels.append(verify(parse(answer), parse(gt)))

        #     all_outputs.append(outputs)
        #     all_labels.append(labels)
        BATCH_SIZE = 1024
        progress_file = f"{args.dataset_path}{args.dataset}_{model_short}_rewards{"_"+str(args.shard) if args.shard is not None else ""}.jsonl"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    all_outputs.append(item['outputs'])
                    all_labels.append(item['labels'])

        # Process in chunks
        for i in tqdm(range(len(all_outputs), len(sgl_dataset), BATCH_SIZE)):
            batch_subset = sgl_dataset[i : i + BATCH_SIZE]
            
            # Run SGLang on the current chunk
            states = compute_partial_answers.run_batch(
                batch_subset,
                num_threads=256
            )

            batch_results = []

            for j, state in enumerate(states):
                # We use i + j to get the correct absolute index from sgl_dataset
                absolute_idx = i + j
                num_steps = len(sgl_dataset[absolute_idx]['CoT'].split('\n\n')) + 1
                
                outputs = []
                labels = []
                
                for step_idx in range(num_steps):
                    raw_output = state[f'answer_{step_idx}']
                    outputs.append(raw_output)
                    
                    # Processing logic
                    answer = '\(' + extract_answer(raw_output) + '\)'
                    gt = '\(' + ds[int(sgl_dataset[absolute_idx]['idx'])]['ground_truth'] + '\)'
                    labels.append(verify(parse(answer), parse(gt)))

                # Create a result object for this specific item
                result_item = {
                    "idx": sgl_dataset[absolute_idx]['idx'],
                    "outputs": outputs,
                    "labels": labels
                }
                batch_results.append(result_item)
                
                # Keep these in memory too if you need them for immediate post-processing
                all_outputs.append(outputs)
                all_labels.append(labels)

            # --- SAVE PROGRESS TO DISK ---
            with open(progress_file, 'a') as f:
                for item in batch_results:
                    f.write(json.dumps(item) + '\n')
                f.flush()

        ds = ds.select(final_ids)
        ds = ds.add_column("CoT", all_CoTs)
        ds = ds.add_column("outputs", all_outputs)
        ds = ds.add_column("labels", all_labels)

        ds.save_to_disk(f"{args.dataset_path}{args.dataset}_{model_short}")

    if args.compute_model_arguments:

        ds = load_from_disk(f"{args.dataset_path}{args.dataset}_{model_short}")

        ds = ds.map(functools.partial(compute_model_arguments, lm_tokenizer=tokenizer, model_name=model_short), num_proc=19)
        ds = ds.filter(lambda row: len(row['indexes'])==len(row['labels']))
        ds.save_to_disk(f"{args.dataset_path}{args.dataset}_{model_short}_corrected")
        
        print("Baseline token count:", sum(ds['token_count'])/len(ds))
        print("Baseline accuracy:", sum([int(labels[-1]) for labels in ds['labels']])/len(ds))
        print("Baseline no thinking accuracy:", sum([int(labels[0]) for labels in ds['labels']])/len(ds))
        ds = ds.filter(lambda example: example['labels'][-1] is True)
        print("Theoretical optimal number of tokens:", sum([datapoint['indexes'][datapoint['labels'].index(True)]-datapoint['indexes'][0] for datapoint in ds]) / len(ds))
        print("Baseline length for only correct answers:", sum(ds['token_count'])/len(ds))