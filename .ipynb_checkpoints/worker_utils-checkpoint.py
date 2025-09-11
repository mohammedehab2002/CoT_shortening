from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from math_verify import parse, verify
from tqdm import tqdm

def get_reward(datapoint, model, tokenizer):

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
    
    for idx, CoT_paragraph in enumerate(["<think>"]+CoT.split('\n\n')):

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

        if idx:
            partial_CoT += '\n\n'

        #print({"answer":answer, "reward": rewards[-1]})

    #print(outputs)
    #print(rewards)

    return outputs, rewards
    
def worker_process(gpu_id, task_queue, results_queue):
    device = f'cuda:{gpu_id}'
    print(f"Worker process started for {device}")
    
    model_name = "a-m-team/AM-Thinking-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",        # FP16 or BF16 depending on hardware
        device_map=None,          # distributes across available GPUs/CPU
        do_sample=False
    ).to(device)
    model.eval()
    
    with torch.no_grad():
        while True:
            task = task_queue.get()
            if task is None:
                print(f"Worker on {device} received exit signal.")
                break
            idx, datapoint = task
            results_queue.put((idx, *get_reward(datapoint, model, tokenizer)))