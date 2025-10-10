import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from architectures import LinearStoppingPolicy
from tqdm import tqdm

def process_datapoint(datapoint, rewards, model):

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

    prod_prob = 1
    expected_reward = 0

    grad = torch.zeros((sum([p.numel() for p in model.grad_params]),), dtype = model.dtype, device=model.device)
    acc_grad = grad.clone()
    
    for idx, CoT_paragraph in enumerate(["<think>"]+CoT.split('\n\n')):

        partial_CoT += CoT_paragraph
        
        text = model.lm_tokenizer.apply_chat_template(
            messages+[{'role': 'assistant', 'content': partial_CoT}],
            tokenize=False,
            add_generation_prompt=True
        )
        text=text[:text.rfind('<|im_end|>')]

        probs = model(text).squeeze()

        stop_grad = torch.autograd.grad(probs[1], model.grad_params, retain_graph=True)
        stop_grad = torch.cat([g.contiguous().view(-1) for g in stop_grad])

        cur_length = model.lm_tokenizer([partial_CoT], return_tensors = "pt").input_ids.shape[1]
        reward = int(rewards[idx]) - cur_length / tot_length

        cur_term = prod_prob * probs[1].detach() * (acc_grad + stop_grad) * reward
        grad += cur_term

        expected_reward += prod_prob * probs[1].detach().item() * reward
        prod_prob *= probs[0].detach().item()

        continue_grad = torch.autograd.grad(probs[0], model.grad_params, retain_graph=False)
        continue_grad = torch.cat([g.contiguous().view(-1) for g in continue_grad])
        acc_grad += continue_grad

        if idx:
            partial_CoT += '\n\n'

    return expected_reward, grad

def set_grad(model, grad):

    pointer = 0
    for p in model.grad_params:
        num_param = p.numel()
        p.grad = grad[pointer:pointer+num_param].view_as(p).clone()
        pointer += num_param

def worker_process(gpu_id, task_queue, grad_update_queue, results_queue, architecture):

    device = f'cuda:{gpu_id}'
    print(f"Worker process started for {device}")
    
    model_name = "a-m-team/AM-Thinking-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    lm = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",        # FP16 or BF16 depending on hardware
        device_map=None,          # distributes across available GPUs/CPU
        do_sample=False
    ).to(device)
    lm.eval()

    model = architecture(lm, tokenizer)

    optim = torch.optim.Adam(model.grad_params, lr=1e-5)
    
    while True:
        if not grad_update_queue.empty():
            grad = grad_update_queue.get_nowait()
            if grad is None:
                print(f"Worker on {device} received exit signal.")
                break
            if isinstance(grad, int):
                torch.save(model.policy_head.state_dict(),f"checkpoints/linear_policy_checkpoint_{grad}.pth")
                print(f"Worker on {device} saved model checkpoint.")
                continue
            set_grad(model, -grad.to(device)) # Negative because we do gradient ascent
            optim.step()
            continue

        task = task_queue.get()
        idx, datapoint, rewards = task
        results_queue.put((idx, *process_datapoint(datapoint, rewards, model)))