# models/distillbert_metagraph.py
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, AutoConfig
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("distillbert_metagraph")
class DistilBertMetaGraph(nn.Module):
    """
    DistilBERT wrapper for MetaGraph pipeline.
    - text_task: "cls" (sentence classification) or "tokcls" (token classification)
    - Returns dict with 'logits' to be robust for different training harnesses.
    - Builds attention_mask internally if not provided (pad_id=0 by default).
    """
    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 pad_id: int = 0,
                 text_task: str = "cls",
                 text_num_classes: int = 2,
                 tok_num_classes: int = 9):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.cfg)
        self.pad_id = pad_id
        self.text_task = text_task
        hid = self.cfg.hidden_size
        if text_task == "cls":
            self.head = nn.Linear(hid, text_num_classes)
        elif text_task == "tokcls":
            self.head = nn.Linear(hid, tok_num_classes)
        else:
            raise ValueError(f"Unsupported text_task={text_task}")

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs):
        # Accept positional call model(x) via kwargs 'x' if needed
        x = input_ids if input_ids is not None else kwargs.get("x")
        assert x is not None, "input_ids must be provided"
        if x.dtype not in (torch.int64, torch.int32):
            raise AssertionError("expect input_ids LongTensor")
        if attention_mask is None:
            attention_mask = (x != self.pad_id).long()

        out = self.bert(input_ids=x, attention_mask=attention_mask)

        if self.text_task == "cls":
            last = out.last_hidden_state  # [B,T,H]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            logits = self.head(pooled)  # [B,C]
        else:  # tokcls
            last = out.last_hidden_state
            logits = self.head(last)    # [B,T,C]
        return {"logits": logits}
