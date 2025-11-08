import torch
import torch.nn as nn

class LinearStoppingPolicy(nn.Module):
    def __init__(
        self,
        lm,
        lm_tokenizer,
        device,
    ):
        super().__init__()
        
        self.lm_tokenizer = lm_tokenizer
        self.lm = lm

        self.device = device
        if lm:
            self.dtype = lm.dtype
        else:
            self.dtype = torch.bfloat16

        # lm_hidden = self.lm.config.hidden_size
        lm_hidden = 5120

        self.policy_head = nn.Linear(lm_hidden, 2).to(dtype=self.dtype,device=self.device)

        self.grad_params = list(self.policy_head.parameters())
        if self.lm:
            self.lm.requires_grad_(False)

        # Initialize policy_head parameters with fixed values
        # with torch.no_grad():
        #     self.policy_head.weight.copy_(torch.zeros_like(self.policy_head.weight, device=self.device, dtype=self.dtype))
        #     self.policy_head.bias.copy_(torch.tensor([0,-4.40368125271], device=self.device, dtype=self.dtype))
        
        self.compute_trainable_state_dict()

    def forward(self, x):

        if isinstance(x, str):

            # x = "a" * 100000

            model_inputs = self.lm_tokenizer([x], return_tensors="pt").to(self.lm.device)

            # print(model_inputs.input_ids.shape[1])

            lm_out = self.lm.model(**model_inputs, use_cache=False)
            last_hidden = lm_out.last_hidden_state[:,-1,:]
        
        else:

            last_hidden = x.to(self.device)

        logits = self.policy_head(last_hidden)

        return logits

    def compute_trainable_state_dict(self):
        # 1. Get the names of all trainable parameters
        trainable_param_names = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_param_names.add(name)

        # 2. Filter the full state_dict to get only the trainable ones
        full_state_dict = self.state_dict()
        self.trainable_state_dict = {name: param for name, param in full_state_dict.items() 
                                if name in trainable_param_names}
    
class TransformerStoppingPolicy(LinearStoppingPolicy):
    def __init__(
        self,
        lm,
        lm_tokenizer,
        device,
        num_finetuning_layers=2,
    ):
        super().__init__(lm, lm_tokenizer, device)
        for layer_num in range(num_finetuning_layers):
            self.lm.model.layers[-layer_num-1].requires_grad_(True)
            self.grad_params.extend(self.lm.model.layers[-layer_num-1].parameters())
        
        self.compute_trainable_state_dict()


# class BertStoppingPolicy(nn.Module):
#     def __init__(
#         self,
#         lm,
#         lm_tokenizer,
#         bert_name="bert-base-uncased",
#     ):
#         super().__init__()
        
#         self.lm_tokenizer = lm_tokenizer
#         self.lm = lm

#         self.dtype = lm.dtype
#         self.device = lm.device

#         # --- load BERT ---
#         # self.bert = BertModel.from_pretrained(bert_name, low_cpu_mem_usage=True).to(dtype=lm.dtype,device=lm.device)
#         # self.bert_config = self.bert.config
#         # bert_hidden = self.bert_config.hidden_size

#         # --- projection from LM hidden dim -> BERT hidden dim ---
#         # lm_hidden = self.lm.config.hidden_size
#         lm_hidden = 5120
#         # self.project = nn.Linear(lm_hidden, bert_hidden).to(dtype=lm.dtype,device=lm.device)

#         # --- policy head ---
#         self.policy_head = nn.Linear(lm_hidden, 2).to(dtype=self.dtype,device=self.device)

#         # self.cls_embed = nn.Parameter(torch.randn(1,1,bert_hidden).to(dtype=lm.dtype,device=lm.device))

#         self.grad_params = list(self.policy_head.parameters())

#     def forward(self, partial_CoT):

#         model_inputs = self.lm_tokenizer([partial_CoT], return_tensors="pt").to(self.lm.device)

#         # with torch.no_grad():

#             # lm_out = self.lm(**model_inputs, output_hidden_states=True)
#             # print(lm_out.hidden_states[-1])
#             # print(self.lm.model(**model_inputs))
#             # last_hidden = lm_out.hidden_states[-1]

#         last_hidden = torch.randn((1,511,5120), device=self.lm.device, dtype=self.lm.dtype)

#         # projected = self.project(last_hidden)

#         # projected_with_cls = torch.cat([self.cls_embed, projected], dim=1)

#         # bert_out = self.bert(
#         #     inputs_embeds=projected_with_cls,
#         #     attention_mask=torch.ones((1,projected_with_cls.shape[1]), device=self.lm.device),
#         #     return_dict=True
#         # )

#         # bert_seq = bert_out.last_hidden_state

#         logits = self.policy_head(last_hidden[:,-1,:])

#         return torch.softmax(logits, dim = -1)
