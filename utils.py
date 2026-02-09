import torch
import numpy as np

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

def compute_metrics_one_datapoint(logits, indexes, labels, config):

    is_numpy = isinstance(logits, np.ndarray)
    if is_numpy:
        logits = torch.from_numpy(logits)

    if config.loss.type == "optimal_stopping_softmax":
        probs = torch.softmax(logits, dim=-1)
    else:
        probs = torch.sigmoid(logits)

    if config.loss.type == "answer_convergence":
        ret = {}
        for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            mask = probs >= thresh
            idx = list(mask).index(True) if True in mask else len(probs) - 1
            ret[f"expected_accuracy_{thresh}"] = int(labels[idx])
            ret[f"expected_length_{thresh}"] = (indexes[idx] - indexes[0])
            ret[f"expected_length_reduction_{thresh}"] = 1 - ((indexes[idx] - indexes[0]) / (indexes[-1] - indexes[0]))
        return ret

    expected_accuracy = torch.tensor(0.0, device=logits.device)
    expected_length = torch.tensor(0.0, device=logits.device)
    expected_length_percentage = torch.tensor(0.0, device=logits.device)
    con_prob = torch.tensor(1.0, device=logits.device)

    for j in range(len(probs)):
        if config.loss.type == "optimal_stopping_softmax":
            stop_prob = probs[j]
        else:
            if j == len(probs) - 1:
                stop_prob = con_prob
            else:
                if config.loss.type == "optimal_stopping":
                    stop_prob = con_prob * (1 - probs[j])
                elif config.loss.type == "optimal_stopping_pre":
                    stop_prob = con_prob - torch.min(con_prob, probs[j])
        expected_accuracy += stop_prob * int(labels[j])
        expected_length += stop_prob * (indexes[j] - indexes[0])
        expected_length_percentage += stop_prob * ((indexes[j] - indexes[0]) / (indexes[-1] - indexes[0]))
        if config.loss.type == "optimal_stopping":
            con_prob = con_prob * probs[j]
        elif config.loss.type == "optimal_stopping_pre":
            con_prob = torch.min(con_prob, probs[j])

    results = {
        "expected_accuracy": expected_accuracy,
        "expected_length": expected_length,
        "expected_length_reduction": 1 - expected_length_percentage,
    }

    if is_numpy:
        results = {k: v.item() for k, v in results.items()}
    
    return results

def compute_metrics(eval_pred, config):

    preds, labels = eval_pred
    logits, indexes, lens = preds

    cur_idx = 0
    aggregated_metrics = {}

    for i in range(len(lens)):

        cur_logits = logits[cur_idx:cur_idx+lens[i]]
        cur_indexes = indexes[cur_idx:cur_idx+lens[i]]
        cur_rewards = labels[i]

        metrics = compute_metrics_one_datapoint(cur_logits, cur_indexes, cur_rewards, config)

        cur_idx += lens[i]
        for k, v in metrics.items():
            aggregated_metrics[k] = aggregated_metrics.get(k, 0) + v
    
    for k in aggregated_metrics.keys():
        aggregated_metrics[k] /= len(lens)
    
    return aggregated_metrics