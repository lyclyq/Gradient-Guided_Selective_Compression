
import torch.nn as nn
from .small_transformer import PatchEmbed, Encoder

class TransformerBaseline(nn.Module):
    def __init__(self, in_ch=1, emb=96, patch=7, img=28, heads=6, mlp=192, depth=4, num_classes=10):
        super().__init__()
        self.pe = PatchEmbed(in_ch, emb, patch, img)
        self.enc = Encoder(emb, heads, mlp, depth, num_classes)
    def groups(self):
        groups = []
        gid = 0
        for m in self.modules():
            ps = [p for p in getattr(m, "parameters", lambda:[])() if p.requires_grad]
            if ps:
                groups.append((f"m{gid}", ps))
                gid += 1
        return groups
    def forward(self, x):
        return self.enc(self.pe(x))
