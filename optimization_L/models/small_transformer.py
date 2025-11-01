
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, emb=64, patch=7, img=28):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, emb, kernel_size=patch, stride=patch)
        self.nh = img // patch
        self.nw = img // patch
        self.num_tokens = self.nh * self.nw
    def forward(self, x):
        x = self.proj(x)          # [B, emb, nh, nw]
        x = x.flatten(2).transpose(1,2)  # [B, T, emb]
        return x

class Encoder(nn.Module):
    def __init__(self, dim=64, heads=4, mlp=128, depth=2, num_classes=10):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.cls = nn.Linear(dim, num_classes)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        h = self.encoder(x)
        h = self.norm(h.mean(dim=1))
        return self.cls(h)

class SmallViT(nn.Module):
    def __init__(self, in_ch=1, emb=64, patch=7, img=28, heads=4, mlp=128, depth=2, num_classes=10):
        super().__init__()
        self.pe = PatchEmbed(in_ch, emb, patch, img)
        self.enc = Encoder(emb, heads, mlp, depth, num_classes)
    def groups(self):
        # define simple per-layer groups for GVA
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
