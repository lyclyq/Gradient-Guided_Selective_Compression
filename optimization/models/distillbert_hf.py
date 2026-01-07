# models/distillbert_hf.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("distillbert_hf")
class DistilBertHF(nn.Module):
    """
    轻量 HF DistilBERT 包装：
      - 文本分类 textcls：logits [B, C]（masked mean-pool）
      - 序列标注 tokcls：logits [B, T, C]
    注意：
      - 只依赖 input_ids(LongTensor)；attention_mask 在 forward 中由 pad_id 自动构造
    """
    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 pad_id: int = 0,
                 text_task: str = "cls",           # "cls" | "tokcls"
                 text_num_classes: int = 2,        # for textcls
                 tok_num_classes: int = 9):        # for tokcls
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

    def forward(self, x: torch.Tensor):
        assert x.dtype in (torch.int32, torch.int64), "expect input_ids LongTensor"
        input_ids = x
        attention_mask = (input_ids != self.pad_id).long()
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.text_task == "cls":
            # DistilBERT 无 pooler：用 masked mean-pool（更稳）
            last = out.last_hidden_state  # [B,T,H]
            mask = attention_mask.unsqueeze(-1).float()       # [B,T,1]
            summed = (last * mask).sum(dim=1)                 # [B,H]
            denom = mask.sum(dim=1).clamp_min(1.0)            # [B,1]
            pooled = summed / denom
            return self.head(pooled)                          # [B,C]

        else:  # tokcls
            last = out.last_hidden_state                      # [B,T,H]
            return self.head(last)                            # [B,T,C]
