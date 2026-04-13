import torch
import torch.nn as nn
from utils import compute_metrics_one_datapoint

class TransformerStoppingPolicy(nn.Module):
    def __init__(
        self,
        lm,
        lm_tokenizer,
        config,
    ):
        super().__init__()
        
        self.lm_tokenizer = lm_tokenizer
        self.lm = lm

        self.dtype = lm.dtype
        lm_hidden = self.lm.config.hidden_size

        self.policy_head = nn.Linear(lm_hidden, 1)
        
        self.lm.requires_grad_(False)
        for layer_num in range(config.num_finetuning_layers):
            self.lm.model.layers[-layer_num-1].requires_grad_(True)

        # Initialize policy_head parameters with fixed values
        if "initialization" in config and config.initialization == "constant":
            with torch.no_grad():
                self.policy_head.weight.copy_(torch.zeros_like(self.policy_head.weight, dtype=self.dtype))
                self.policy_head.bias.copy_(torch.tensor([4.40368125271], dtype=self.dtype))

        self.cfg = config
        self.device = self.lm.device

    def infer(self, input_ids, attention_mask, indexes):

        lm_out = self.lm.model(input_ids, attention_mask, use_cache=False)
        last_hidden = lm_out.last_hidden_state[:,indexes,:]

        logits = self.policy_head(last_hidden)

        return logits.squeeze(0).squeeze(-1)
    
    def forward(self, input_ids, attention_mask, indexes, labels):

        indexes = indexes.squeeze(0)
        labels = labels.squeeze(0)

        logits = self.infer(input_ids, attention_mask, indexes)

        if self.cfg.loss.type == "logistic_regression" or self.cfg.loss.type == "answer_convergence":
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            tot_loss = loss_fn(logits, labels.to(logits.dtype)).mean()
        else:
            metrics = compute_metrics_one_datapoint(logits, indexes, labels, self.cfg)
            if self.cfg.loss.normalize:
                tot_loss = - (metrics['expected_accuracy'] + self.cfg.loss.lam * metrics['expected_length_reduction'] - 1)
            else:
                tot_loss = - (metrics['expected_accuracy'] - self.cfg.loss.lam * metrics['expected_length'])

        return tot_loss, logits, indexes, torch.tensor(len(indexes))
    
    def state_dict(self, *args, **kwargs):

        full_state_dict = super().state_dict(*args, **kwargs)
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in full_state_dict.items() if k in trainable_names}