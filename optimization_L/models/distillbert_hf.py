
import torch
import torch.nn as nn
from transformers import AutoModel

class DistillBertClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hid = self.backbone.config.hidden_size
        self.head = nn.Linear(hid, num_labels)

    def groups(self):
        groups = []
        gid = 0
        for m in self.modules():
            ps = [p for p in getattr(m, "parameters", lambda:[])() if p.requires_grad]
            if ps:
                groups.append((f"m{gid}", ps))
                gid += 1
        return groups

    def forward(self, input_ids=None, attention_mask=None, **kw):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw)
        h = out.last_hidden_state[:,0,:]
        return self.head(h)
