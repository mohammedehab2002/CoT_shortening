import torch
from architectures import LinearStoppingPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import numpy as np

def process_datapoint(datapoint, rewards, model, loss_type):

    messages = []

    if datapoint['system'] is None:
        datapoint['system'] = ""
    datapoint['system'] += "The answer should just be your final answer in \\boxed{} without any explanation or reasoning between the <answer> and </answer> tags. Even if you're not sure, only give me the final answer."

    if datapoint['system'] is not None:
        messages.append({'role': 'system', 'content': datapoint['system']})

    messages.append({'role': 'user', 'content': datapoint['conversations'][0]['value']})
    
    CoT = datapoint['conversations'][1]['value']
    tot_length = model.lm_tokenizer([CoT], return_tensors = "pt").input_ids.shape[1]
    CoT = CoT[7:CoT.find('<answer>')]
    partial_CoT = ""

    grad = torch.zeros((sum([p.numel() for p in model.grad_params]),), dtype = model.dtype, device=model.device)
    acc_grad = grad.clone()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    tot_paragraphs = len(CoT.split('\n\n')) + 1

    tot_loss = 0.0
    expected_accuracy = 0.0
    expected_length = 0.0

    prod_prob = 1
    
    for idx, CoT_paragraph in enumerate(["<think>"]+CoT.split('\n\n')):

        partial_CoT += CoT_paragraph

        if 'hidden_states' in datapoint:
            logits = model(datapoint['hidden_states'][idx,:].to(model.dtype)).squeeze()
        else:
            text = model.lm_tokenizer.apply_chat_template(
                messages+[{'role': 'assistant', 'content': partial_CoT}],
                tokenize=False,
                add_generation_prompt=True
            )
            text=text[:text.rfind('<|im_end|>')]

            logits = model(text).squeeze()

        if loss_type == "logistic_regression":

            target = torch.zeros((2,), device = model.device, dtype = model.dtype)
            target[int(rewards[idx])] = 1
            
            loss = loss_fn(logits, target) / tot_paragraphs

            tot_loss += loss.item()

            # print("before backward", model.device, torch.cuda.max_memory_allocated(model.device) / 1024 ** 3)

            cur_grad = torch.autograd.grad(loss, model.grad_params, retain_graph=False)
            # print("after backward", model.device, torch.cuda.max_memory_allocated(model.device) / 1024 ** 3)
            cur_grad = torch.cat([g.contiguous().view(-1) for g in cur_grad])
            # print("after concat", model.device, torch.cuda.max_memory_allocated(model.device) / 1024 ** 3)
            grad += cur_grad
            del cur_grad

        else:

            probs = torch.softmax(logits, dim = -1)

            stop_grad = torch.autograd.grad(torch.log(probs[1]), model.grad_params, retain_graph=True)
            stop_grad = torch.cat([g.contiguous().view(-1) for g in stop_grad])

            cur_length = model.lm_tokenizer([partial_CoT], return_tensors = "pt").input_ids.shape[1]
            loss = cur_length / tot_length - int(rewards[idx])

            stop_prob = prod_prob * probs[1].detach().item()
            if idx == tot_paragraphs - 1:
                stop_prob = prod_prob

            cur_term = stop_prob * (acc_grad + stop_grad) * loss
            grad += cur_term

            tot_loss += stop_prob * loss
            expected_accuracy += stop_prob * int(rewards[idx])
            expected_length += stop_prob * cur_length / tot_length
            prod_prob *= probs[0].detach().item()

            continue_grad = torch.autograd.grad(torch.log(probs[0]), model.grad_params, retain_graph=False)
            continue_grad = torch.cat([g.contiguous().view(-1) for g in continue_grad])
            acc_grad += continue_grad

        if idx:
            partial_CoT += '\n\n'

    cpu_grad = grad.cpu()
    del grad
    torch.cuda.empty_cache()

    return tot_loss, cpu_grad, expected_accuracy, expected_length

def set_grad(model, grad):

    pointer = 0
    for p in model.grad_params:
        num_param = p.numel()
        p.grad = grad[pointer:pointer+num_param].view_as(p).clone()
        pointer += num_param

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_process(gpu_id, task_queue, grad_update_queue, results_queue, loss_type, policy_class, load_hidden_states):

    set_seed(42)

    device = f'cuda:{gpu_id}'
    print(f"Worker process started for {device}")

    model_name = "a-m-team/AM-Thinking-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if load_hidden_states:
        lm = None
    else:
        
        lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=None,
            do_sample=False
        ).to(device)
        lm.eval()

    model = policy_class(lm, tokenizer, device)
    
    while True:
        if not grad_update_queue.empty():
            upd = grad_update_queue.get_nowait()
            if upd is None:
                print(f"Worker on {device} received exit signal.")
                break
            checkpoint = torch.load(
                upd,
                map_location=f"cuda:{gpu_id}"
            )['model']
            for k, v in checkpoint.items():
                checkpoint[k] = v.to(dtype=torch.bfloat16)
            model.load_state_dict(checkpoint, strict=False)
            print("gpu checksum:", sum(p.to(torch.float32).data.sum() for p in model.grad_params))

        if not task_queue.empty():
            if not grad_update_queue.empty():
                continue
            task = task_queue.get()
            idx, datapoint, rewards = task
            results_queue.put((idx, *process_datapoint(datapoint, rewards, model, loss_type)))