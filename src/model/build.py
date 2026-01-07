import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from .adapters import AdapterLinear

def inject_adapters(model, r:int, r_alt:int, use_alt:bool, ab_only:bool=False):
    """Replace selected Linear layers with AdapterLinear. Base model params frozen."""
    for p in model.parameters():
        p.requires_grad_(False)

    def replace_linear(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                wrapped = AdapterLinear(child, r=r, r_alt=r_alt, use_alt=(use_alt and (not ab_only)))
                setattr(module, name, wrapped)
            else:
                replace_linear(child)

    replace_linear(model.bert.encoder)  # focus on encoder linears
    # classifier head stays trainable? For fairness, keep head trainable (small)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    return model

def build_model(model_name:str, num_labels:int=2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
