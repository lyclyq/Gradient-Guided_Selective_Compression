
import torch
import torch.nn as nn
from transformers import AutoModel

class SmallBertClassifier(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-tiny", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hid = self.bert.config.hidden_size
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
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kw)
        # Use [CLS]
        if hasattr(out, "last_hidden_state"):
            cls = out.last_hidden_state[:,0,:]
        else:
            cls = out[0][:,0,:]
        return self.head(cls)
