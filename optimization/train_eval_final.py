# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Final train_eval.py (PCA/FFT denoise + OverfitGuard + robust text batch handling).
# - Robust batch parsing: dict / (x,y) / (x,mask,y) / ((x,mask),y) / raw text
# - Auto-tokenize raw text to BERT ids when needed (bert-base-uncased)
# - Strictly denoise-only (no train→valid compensation)
# - SwanLab logging + CSV logging
# """

# import os, argparse, random
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from datasets import load_dataset

# # ---------------------------
# # Globals for logging backends
# # ---------------------------
# swan = None     # SwanLab run handle
# csvlog = None   # CSV logger handle

# # --- optional: transformers tokenizer for raw text ---
# _TOKENIZER = None
# def _get_tokenizer():
#     global _TOKENIZER
#     if _TOKENIZER is None:
#         try:
#             from transformers import AutoTokenizer
#             _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
#         except Exception as e:
#             raise RuntimeError(
#                 f"Need transformers to tokenize raw text for small_bert. Install it or "
#                 f"provide a model.collate_fn. Error: {e}"
#             )
#     return _TOKENIZER

# # --- DCA & friends (robust imports) ---
# try:
#     from modules.dca import DCACfg, DCA
# except Exception as e:
#     DCACfg = None; DCA = None
#     print(f"[WARN] DCA unavailable: {e}")

# try:
#     from modules.dca_derivative import DCADerivative
# except Exception as e:
#     DCADerivative = None
#     print(f"[WARN] DCADerivative unavailable: {e}")

# try:
#     from modules.dca_e_ctr_resmix import build_e_ctr_resmix
# except Exception as e:
#     build_e_ctr_resmix = None
#     print(f"[WARN] build_e_ctr_resmix unavailable: {e}")

# try:
#     from modules.spec_reg import spectral_penalty_depthwise
# except Exception:
#     spectral_penalty_depthwise = None

# from modules.soft_resmix import SoftResMixNudger
# from modules.pca_overlap import PCAOverlapManager
# from modules.overfit_guard import OverfitGuard, OverfitGuardConfig

# # --- models ---
# from models.distillbert_hf_gate import DistillBertHFGate
# from models.small_bert import SmallBERT
# from models.small_transformer import SmallTransformer
# # 可选：baseline（无插桩）对比
# # from models.distillbert_hf import DistilBertHF

# MODEL_MAP = {
#     'distillbert_hf_gate': DistillBertHFGate,
#     'small_bert': SmallBERT,
#     'small_transformer': SmallTransformer,
#     # 'distillbert_hf': DistilBertHF,  # 可选：baseline
# }

# def set_seed(seed=42):
#     random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# def build_sst2_loaders(batch, collate_fn=None):
#     ds = load_dataset("glue", "sst2")
#     train = list(zip(ds['train']['sentence'], ds['train']['label']))
#     val   = list(zip(ds['validation']['sentence'], ds['validation']['label']))
#     return DataLoader(train, batch_size=batch, shuffle=True, collate_fn=collate_fn, drop_last=True), \
#            DataLoader(val,   batch_size=batch, shuffle=False, collate_fn=collate_fn)

# def accuracy(logits, labels):
#     preds = logits.argmax(-1)
#     return (preds == labels).float().mean().item()

# # ---------------------------
# # Robust batch parser + auto-tokenize
# # ---------------------------
# def _maybe_tokenize_text(x_like, max_len, device):
#     """
#     If x_like is list[str] or str -> tokenize to (input_ids, attention_mask)
#     Else: return (None, None) to indicate 'no tokenization was done'
#     """
#     # list/tuple of strings
#     if isinstance(x_like, (list, tuple)) and len(x_like) > 0 and isinstance(x_like[0], str):
#         tok = _get_tokenizer()
#         enc = tok(
#             list(x_like),
#             padding='max_length',       # fixed len
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt"
#         )
#         return enc["input_ids"].to(device), enc["attention_mask"].to(device)

#     # single string (unlikely in a batch)
#     if isinstance(x_like, str):
#         tok = _get_tokenizer()
#         enc = tok(
#             [x_like],
#             padding='max_length',       # fixed len
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt"
#         )
#         return enc["input_ids"].to(device), enc["attention_mask"].to(device)

#     return None, None

# def parse_batch(batch, device, max_len=128):
#     """
#     Normalize batch to (x_ids, attn_mask, labels).

#     Supports:
#       - dict: {'input_ids','attention_mask','labels'} or {'x':(ids,mask),'labels':y}
#       - (x, y)
#       - (x, mask, y)
#       - ((x, mask), y)
#       - ( {'input_ids':..., 'attention_mask':...}, y )
#       - raw text: list[str] or str  -> auto tokenize to ids/mask
#     """
#     def to_dev(t):
#         return t.to(device) if torch.is_tensor(t) else t

#     def norm_x(xlike):
#         # 1) dict path
#         if isinstance(xlike, dict):
#             xi = xlike.get('input_ids', xlike.get('x', xlike.get('inputs')))
#             xm = xlike.get('attention_mask', xlike.get('mask'))
#             # auto tokenize if raw text
#             ids, am = _maybe_tokenize_text(xi, max_len, device)
#             if ids is not None:
#                 return ids, am
#             # handle xi as (ids,mask)
#             if (xm is None) and isinstance(xi, (tuple, list)) and len(xi) == 2:
#                 xi, xm = xi
#             return to_dev(xi), to_dev(xm)

#         # 2) tuple/list path
#         if isinstance(xlike, (tuple, list)):
#             # common: (ids, mask)
#             if len(xlike) == 2 and torch.is_tensor(xlike[0]):
#                 return to_dev(xlike[0]), to_dev(xlike[1])
#             # raw text list[str]
#             ids, am = _maybe_tokenize_text(xlike, max_len, device)
#             if ids is not None:
#                 return ids, am
#             # fallback: take first as ids
#             if len(xlike) >= 1:
#                 return to_dev(xlike[0]), None

#         # 3) single tensor -> ids only
#         if torch.is_tensor(xlike):
#             return to_dev(xlike), None

#         # 4) raw text string
#         ids, am = _maybe_tokenize_text(xlike, max_len, device)
#         if ids is not None:
#             return ids, am

#         # unknown structure
#         return xlike, None

#     # --- dict batch ---
#     if isinstance(batch, dict):
#         xi, xm = norm_x(batch)
#         y = batch.get('labels', batch.get('y', None))
#         return xi, xm, to_dev(y)

#     # --- tuple/list batch ---
#     if isinstance(batch, (tuple, list)):
#         if len(batch) == 2:
#             a, b = batch
#             xi, xm = norm_x(a)
#             y = b
#             return xi, xm, to_dev(y)
#         if len(batch) == 3:
#             xi, xm = norm_x(batch[0])
#             xm = batch[1] if (xm is None) else xm
#             y = batch[2]
#             return xi, to_dev(xm), to_dev(y)
#         # other lengths: best effort
#         xi, xm = norm_x(batch[0])
#         if xm is None and len(batch) > 2:
#             xm = batch[1]
#         y = batch[-1]
#         return xi, to_dev(xm), to_dev(y)

#     # --- single tensor ---
#     if torch.is_tensor(batch):
#         return batch.to(device), None, None

#     # --- raw text fallthrough ---
#     ids, am = _maybe_tokenize_text(batch, max_len, device)
#     if ids is not None:
#         return ids, am, None

#     raise TypeError(f"Unsupported batch type: {type(batch)}")

# def call_model(model, x_ids, attn_mask):
#     """
#     Unified call:
#     - DistilBertHFGate expects attention_mask keyword
#     - SmallBERT/SmallTransformer: forward(x, attn_mask=...) or forward(x)
#     """
#     # DistilBERT wrapper里一般有 bert 子模块
#     if hasattr(model, 'bert') or model.__class__.__name__.lower().startswith('distill'):
#         return model(x_ids, attention_mask=attn_mask)
#     # SmallBERT / SmallTransformer
#     try:
#         return model(x_ids, attn_mask=attn_mask) if attn_mask is not None else model(x_ids)
#     except TypeError:
#         # 某些实现forward签名只有 (x)，忽略mask
#         return model(x_ids)

# # ---------------------------
# # Train / Eval
# # ---------------------------
# def run_epoch(model, loader, opt, device, state, args, train=True):
#     model.train(train)
#     tot_loss=tot_acc=n=0
#     for batch in loader:
#         if hasattr(model, 'collate_fn'):
#             batch = model.collate_fn(batch)
#         x_ids, attn_mask, labels = parse_batch(batch, device, max_len=args.max_len)
#         assert torch.is_tensor(x_ids), f"x_ids must be Tensor, got {type(x_ids)}"

#         if train: opt.zero_grad(set_to_none=True)
#         out = call_model(model, x_ids, attn_mask)
#         logits = out['logits']
#         # Guard: ensure correct shape (当前主跑 textcls)
#         assert logits.dim() == 2 and logits.size(1) == args.num_labels, \
#             f"expect logits [B,{args.num_labels}], got {tuple(logits.shape)}"

#         if labels is None:
#             continue
#         loss = out.get('loss') or F.cross_entropy(logits, labels)
#         acc = accuracy(logits, labels)
#         if train:
#             loss.backward(); opt.step(); state['step']+=1
#         bs=labels.size(0); tot_loss+=loss.item()*bs; tot_acc+=acc*bs; n+=bs

#         if train and args.log_every>0 and state['step']%args.log_every==0:
#             mean_loss=tot_loss/max(1,n); mean_acc=tot_acc/max(1,n)
#             print(f"[S{state['step']}] train loss={mean_loss:.4f} acc={mean_acc:.4f}")
#             # step logs to Swan/CSV
#             try:
#                 if swan is not None:
#                     swan.log({"train/step_loss": float(mean_loss),
#                               "train/step_acc": float(mean_acc)}, step=state['step'])
#                 if csvlog is not None:
#                     csvlog.log({"stage":"train_step","step":state['step'],"epoch":-1,
#                                 "loss":round(float(mean_loss),6),"acc":round(float(mean_acc),6)})
#             except Exception:
#                 pass

#     return {'loss':tot_loss/max(1,n),'acc':tot_acc/max(1,n)}

# @torch.no_grad()
# def evaluate(model, loader, device, args):
#     model.eval(); tot_loss=tot_acc=n=0
#     for batch in loader:
#         if hasattr(model, 'collate_fn'):
#             batch = model.collate_fn(batch)
#         x_ids, attn_mask, labels = parse_batch(batch, device, max_len=args.max_len)
#         assert torch.is_tensor(x_ids), f"x_ids must be Tensor, got {type(x_ids)}"
#         out = call_model(model, x_ids, attn_mask)
#         logits = out['logits']
#         assert logits.dim() == 2 and logits.size(1) == args.num_labels, \
#             f"expect logits [B,{args.num_labels}], got {tuple(logits.shape)}"

#         loss = out.get('loss') or F.cross_entropy(logits, labels)
#         acc = accuracy(logits, labels)
#         bs=labels.size(0); tot_loss+=loss.item()*bs; tot_acc+=acc*bs; n+=bs
#     return {'loss':tot_loss/max(1,n),'acc':tot_acc/max(1,n)}

# @torch.no_grad()
# def quickval_arch_step(model, val_loader, device, args, dca, dca_drv, resmix_nudger, pca_mgr, guard):
#     # ---- 1) 收集一个小窗做代理 CE ----
#     it = iter(val_loader)
#     batches = []
#     for _ in range(max(1, args.arch_val_batches)):
#         try:
#             b = next(it)
#         except StopIteration:
#             break
#         if hasattr(model, 'collate_fn'):
#             b = model.collate_fn(b)
#         x_ids, attn_mask, labels = parse_batch(b, device, max_len=args.max_len)
#         batches.append((x_ids, attn_mask, labels))
#     if not batches:
#         return

#     ce_vals = []
#     for x_ids, attn_mask, labels in batches:
#         out = call_model(model, x_ids, attn_mask)
#         logits = out['logits']
#         ce_vals.append(F.cross_entropy(logits, labels))
#     proxy_ce = torch.stack(ce_vals).mean()

#     # ---- 2) DCA / 导数权重（可选）----
#     L_alpha = torch.tensor(0.0, device=device)
#     if args.dca_enable and args.dca_mode == 'e_ctr_resmix' and (dca is not None) and (build_e_ctr_resmix is not None):
#         L_alpha = build_e_ctr_resmix(dca).loss()

#     drv_weight = 1.0
#     if args.dca_derivative_enable and (DCADerivative is not None) and dca_drv is not None:
#         drv_weight = dca_drv.update_and_weight(proxy_ce, mode=args.dca_drv_mode,
#                                                lam=args.dca_drv_lambda, kappa=args.dca_drv_kappa)

#     # ---- 3) 过拟合守卫：是否触发 ----
#     armed, guard_w, extras = guard.tick(train_proxy=float(proxy_ce.item()),
#                                         val_proxy=float(proxy_ce.item()))

#     # ---- 4) 预计算 PCA 门（总是先算，后面看是否采用）----
#     pca_layer_gs = []
#     pca_scores_for_pick = []  # s_pca 候选相似度
#     if pca_mgr is not None:
#         for lid, wrap in getattr(model, 'pca_wrappers', {}).items():
#             Xval = getattr(wrap, '_last_input', None)
#             if Xval is None:
#                 continue
#             scores = pca_mgr.overlap_scores(lid, Xval)  # {'O_i','O_res'}
#             O_i, O_res = scores['O_i'], scores['O_res']
#             g_i  = torch.clamp(1.0 - args.pca_lambda * (1.0 - O_i),  min=args.pca_amin, max=1.0)
#             g_res= torch.clamp(1.0 - args.pca_lambda * (1.0 - O_res), min=args.pca_amin, max=1.0)
#             wrap.cached_gains = {"comp": g_i.detach(), "res": g_res.detach()}  # 先缓存，不立即 set
#             pca_layer_gs.append(float(g_i.mean().item()))
#             # 适配度：s_pca = 1 - O_res（越大越适合 PCA）
#             pca_scores_for_pick.append(float((1.0 - O_res).mean().item()))
#     pca_g_mean = (sum(pca_layer_gs) / len(pca_layer_gs)) if pca_layer_gs else 1.0

#     # ---- 5) 为 FFT / LP 预备相似度 ----
#     s_fft = float(guard_w)   # 越大越适合频域压制
#     s_lp  = 0.5              # 常数先验（可替换为 gate 熵等信号）

#     # ---- 6) 根据策略选择：prob / softmix / deterministic ----
#     # 相似度：
#     s_pca = (sum(pca_scores_for_pick)/len(pca_scores_for_pick)) if pca_scores_for_pick else 0.0
#     s = torch.tensor([s_pca, s_fft, s_lp], device=device, dtype=torch.float32)

#     # 阈值截断：低于阈值的候选直接屏蔽
#     tau = float(getattr(args, 'arm_thr', 0.2))
#     mask = s >= tau
#     if not bool(mask.any()):
#         # 回退：只保留最大者
#         mask = torch.zeros_like(s, dtype=torch.bool)
#         mask[torch.argmax(s)] = True

#     # 反比平方（可配幂次）：w_k ∝ 1 / (1 - s_k + eps)^{arm_invpow}
#     eps = float(getattr(args, 'arm_eps', 1e-6))
#     invpow = float(getattr(args, 'arm_invpow', 2.0))
#     raw_w = torch.zeros_like(s)
#     raw_w[mask] = 1.0 / torch.clamp(1.0 - s[mask] + eps, min=eps).pow(invpow)

#     # 乘以先验权重
#     aw = {'pca': 1.0, 'fft': 1.0, 'lp': 1.0}
#     try:
#         for kv in (args.arm_weights or '').split(','):
#             if '=' in kv:
#                 k, v = kv.split('=', 1)
#                 aw[k.strip().lower()] = float(v)
#     except Exception:
#         pass
#     priors = torch.tensor([aw['pca'], aw['fft'], aw['lp']], device=device, dtype=torch.float32)
#     raw_w = raw_w * priors

#     # 温度化 softmax -> 概率
#     temp = max(1e-6, float(getattr(args, 'arm_temp', 1.0)))
#     probs = torch.softmax(raw_w / temp, dim=0)

#     # 选择策略
#     if args.arm_pick == 'softmix':
#         pick = 'softmix'
#     elif args.arm_pick == 'deterministic':
#         pick = ['pca','fft','lp'][int(torch.argmax(probs).item())]
#     else:  # 'prob'
#         idx = torch.multinomial(probs, 1).item()
#         pick = ['pca','fft','lp'][idx]

#     # ---- 7) 若 armed：按选择执行；否则只做日志 ----
#     if armed:
#         # 7.1 PCA：把缓存的 gains 写入
#         if pick in ('pca', 'softmix'):
#             for lid, wrap in getattr(model, 'pca_wrappers', {}).items():
#                 gains = getattr(wrap, 'cached_gains', None)
#                 if not gains:
#                     continue
#                 wrap.set_gains(lid, gains)

#         # 7.2 FFT：把导数/守卫权重注入 feature mask
#         if pick in ('fft', 'softmix'):
#             for fm in getattr(model, 'scan_feature_fft_wrappers', lambda: [])():
#                 try:
#                     fm.set_external_gate(float(guard_w))
#                 except Exception:
#                     pass

#         # 7.3 LP（DARTS）：在触发时把 tau 冷却一点（更“尖”）
#         if pick in ('lp', 'softmix'):
#             for g in getattr(model, 'scan_gate_modules', lambda: [])():
#                 try:
#                     if hasattr(g, 'tau'):
#                         new_tau = max(getattr(g, 'tau', 1.0) * 0.95, 0.5)
#                         getattr(g, 'set_tau', lambda *_: None)(new_tau)
#                 except Exception:
#                     pass

#     # ---- 8) 打印 & 记录 ----
#     msg = (f"[QUICKVAL]{'[ARMED]' if armed else ''} "
#            f"proxy_ce={proxy_ce.item():.4f} drv_w={float(drv_weight):.3f} "
#            f"pick={pick} "
#            f"L_alpha={(float(L_alpha.item()) if hasattr(L_alpha,'item') else float(L_alpha)):.4f}")
#     print(msg)
#     try:
#         if swan is not None:
#             swan.log({
#                 "quickval/proxy_ce": float(proxy_ce.item()),
#                 "quickval/drv_weight": float(drv_weight),
#                 "quickval/armed": 1.0 if armed else 0.0,
#                 "quickval/pick_pca": float(probs[0].item()),
#                 "quickval/pick_fft": float(probs[1].item()),
#                 "quickval/pick_lp":  float(probs[2].item()),
#                 "quickval/L_alpha": float(L_alpha.item()) if hasattr(L_alpha,'item') else float(L_alpha),
#             })
#         if csvlog is not None:
#             csvlog.log({
#                 "stage": "quickval_armed" if armed else "quickval",
#                 "step": -1, "epoch": -1,
#                 "proxy_ce": round(float(proxy_ce.item()), 6),
#                 "drv_weight": round(float(drv_weight), 6),
#                 "p_pick_pca": round(float(probs[0].item()), 6),
#                 "p_pick_fft": round(float(probs[1].item()), 6),
#                 "p_pick_lp":  round(float(probs[2].item()), 6),
#                 "pick": pick,
#                 "L_alpha": round(float(L_alpha.item()) if hasattr(L_alpha,'item') else float(L_alpha), 6),
#             })
#     except Exception:
#         pass



#         # ---- 8.5) 每层能量统计（供能量引导步长使用）----
#     layer_energy = {}
#     meanE = 0.0
#     if getattr(args, 'energy_guided', False) and hasattr(model, 'pca_wrappers'):
#         energies = []
#         for lid, wrap in getattr(model, 'pca_wrappers', {}).items():
#             X_l = getattr(wrap, '_last_input', None)
#             if X_l is None:
#                 continue
#             # X_l: [B, T, D] -> E_l = mean(||X||_F^2 / (T*D))
#             B, T, D = X_l.shape
#             e = (X_l.pow(2).sum(dim=(1, 2)) / float(T * D)).mean()  # scalar tensor
#             e_val = float(e.item())
#             layer_energy[lid] = e_val
#             energies.append(e_val)
#         if len(energies) > 0:
#             meanE = sum(energies) / len(energies)

#     # ---- 9) 结构 loss 回传（如有）+ ResMix 轻 nudging（可选）----
#     arch_loss = (1.0 + args.dca_beta * float(drv_weight)) * L_alpha
#     if args.spec_reg_enable and (spectral_penalty_depthwise is not None):
#         arch_loss = arch_loss + spectral_penalty_depthwise(model, lam=args.spec_reg_lambda, power=args.spec_reg_power)
#     if arch_loss.requires_grad and float(arch_loss) != 0.0:
#         arch_loss.backward()

#     # if resmix_nudger is not None and args.resmix_from_dca_lr > 0:
#     #     resmix_nudger.nudge(step_size=args.resmix_from_dca_lr * float(drv_weight))

#     if resmix_nudger is not None and args.resmix_from_dca_lr > 0:
#         base = args.resmix_from_dca_lr * float(drv_weight)
#     if not getattr(args, 'energy_guided', False) or len(layer_energy) == 0:
#         # 兼容：不开能量引导 -> 原来的一刀切步长
#         resmix_nudger.nudge(step_size=base)
#     else:
#         # 能量引导：每层一个步长
#         eps   = float(getattr(args, 'energy_eps',   1e-8))
#         gamma = float(getattr(args, 'energy_gamma', 1.0))
#         mE = meanE if meanE > 0.0 else eps

#         step_dict = {}
#         for lid, e in layer_energy.items():
#             # step_l = base * ((E_l / meanE) + eps)^gamma
#             scale = ((e / mE) + eps) ** gamma
#             step_dict[lid] = base * float(scale)

#         # 可选：打印一下本次触发时各层步长
#         try:
#             print(f"[Energy-guided] drv={float(drv_weight):.4f}, meanE={mE:.6f}, "
#                     f"steps={{" + ", ".join([f"{lid}:{step_dict[lid]:.3e}" for lid in sorted(step_dict)]) + "}}")
#         except Exception:
#             pass

#         resmix_nudger.nudge_per_layer(step_dict)


# def main():
#     p=argparse.ArgumentParser()
#     p.add_argument('--task',type=str,default='textcls', choices=['vision','textcls','tokcls'])
#     p.add_argument('--text_dataset',type=str,default='sst2', choices=['sst2','ag_news','trec6','conll2003'])
#     p.add_argument('--max_len',type=int,default=128)  # for tokenizer
#     p.add_argument('--batch',type=int,default=64)
#     p.add_argument('--epochs',type=int,default=10)
#     p.add_argument('--seed',type=int,default=42)
#     p.add_argument('--model',type=str,default='distillbert_hf_gate',
#                   choices=list(MODEL_MAP.keys()))
#     p.add_argument('--num_labels',type=int,default=2)
#     p.add_argument('--insert_layers',type=str,default='last')
#     p.add_argument('--weight_lr',type=float,default=3e-4)
#     p.add_argument('--adam_eps',type=float,default=1e-8)
#     p.add_argument('--adam_wd',type=float,default=0.0)

#     # modules
#     p.add_argument('--gate_enable',action='store_true')
#     p.add_argument('--resmix_enable',action='store_true')
#     p.add_argument('--resmix_from_dca_lr',type=float,default=0.0)

#     # filters / fft
#     p.add_argument('--filter_backend',type=str,default='lp_fixed', choices=['darts','lp_fixed','none'])
#     p.add_argument('--lp_k',type=int,default=5)
#     p.add_argument('--gate_tau',type=float,default=1.0)
#     p.add_argument('--feature_mask', action='store_true')
#     p.add_argument('--ff_gamma',type=float,default=1.0)
#     p.add_argument('--ff_amin',type=float,default=0.8)
#     p.add_argument('--ff_apply_on',type=str,default='input', choices=['input','output'])
#     p.add_argument('--ff_use_drv_gate',action='store_true')

#     # DCA
#     p.add_argument('--dca_enable',action='store_true')
#     p.add_argument('--dca_mode', type=str, default='e_ctr_resmix', choices=['e_ctr_resmix'])
#     p.add_argument('--dca_beta',type=float,default=1.0)
#     p.add_argument('--dca_w_lr',type=float,default=0.0)

#     # derivative gate
#     p.add_argument('--dca_derivative_enable',action='store_true')
#     p.add_argument('--dca_drv_mode',type=str,default='ema', choices=['ema','window'])
#     p.add_argument('--dca_drv_lambda',type=float,default=0.2)
#     p.add_argument('--dca_drv_kappa',type=float,default=5.0)

#     # --- energy-guided step size (optional) ---
#     p.add_argument('--energy_guided', action='store_true',
#                    help='Enable energy-guided per-layer step size for residual mixing.')
#     p.add_argument('--energy_gamma', type=float, default=1.0,
#                    help='Exponent gamma for (E_l / meanE + eps)^gamma.')
#     p.add_argument('--energy_eps', type=float, default=1e-8,
#                    help='Small epsilon to stabilize energy normalization.')

#     # PCA denoise
#     p.add_argument('--pca_enable',action='store_true')
#     p.add_argument('--pca_k',type=int,default=32)
#     p.add_argument('--pca_lambda',type=float,default=0.8)
#     p.add_argument('--pca_amin',type=float,default=0.85)
#     p.add_argument('--pca_patience',type=int,default=2)

#     # quickval 选择策略
#     p.add_argument('--arm_pick', type=str, default='prob', choices=['prob','softmix','deterministic'])
#     p.add_argument('--arm_temp', type=float, default=1.0)
#     p.add_argument('--arm_weights', type=str, default='pca=1,fft=1,lp=1')
#     # ★ 新增：阈值与反比平方
#     p.add_argument('--arm_thr', type=float, default=0.2, help='similarity threshold; below this, candidate is masked out')
#     p.add_argument('--arm_invpow', type=float, default=2.0, help='inverse power for probability weighting (e.g., 2 for inverse-square)')
#     p.add_argument('--arm_eps', type=float, default=1e-6, help='small epsilon to stabilize inverse weighting')

#     # logging & quickval
#     p.add_argument('--log_every',type=int,default=50)
#     p.add_argument('--quickval_every',type=int,default=400)
#     p.add_argument('--arch_val_batches',type=int,default=2)

#     # spec reg (optional)
#     p.add_argument('--spec_reg_enable',action='store_true')
#     p.add_argument('--spec_reg_lambda',type=float,default=1e-4)
#     p.add_argument('--spec_reg_power',type=float,default=1.0)

#     # io
#     p.add_argument('--log_dir',type=str,default='logs')
#     p.add_argument('--run_tag',type=str,default='run')
#     p.add_argument('--save_dir',type=str,default='checkpoints')
#     p.add_argument('--save_name',type=str,default='model.pt')

#     # SwanLab & CSV
#     p.add_argument('--swan_enable', action='store_true')
#     p.add_argument('--swan_project', type=str, default='default')
#     p.add_argument('--swan_experiment', type=str, default='')
#     p.add_argument('--swan_dir', type=str, default='swan_logs')
#     p.add_argument('--csv_log', action='store_true')
#     p.add_argument('--csv_path', type=str, default='optimization/log/metrics.csv')

#     args=p.parse_args()

#     # --- SwanLab init (optional) ---
#     global swan, csvlog
#     if args.swan_enable:
#         try:
#             import swanlab
#             swan = swanlab.init(
#                 project=args.swan_project,
#                 name=(args.swan_experiment or args.run_tag),
#                 work_dir=args.swan_dir,
#                 config=vars(args)
#             )
#             print("[SWAN] logging enabled.")
#         except Exception as e:
#             print(f"[SWAN] init failed: {e}")
#             swan = None

#     # --- CSV logger (optional) ---
#     if args.csv_log:
#         import csv
#         os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
#         class _CSVLogger:
#             def __init__(self, path):
#                 self.path = path
#                 self._header_written = os.path.exists(path) and os.path.getsize(path) > 0
#             def log(self, row: dict):
#                 hdr = list(row.keys())
#                 need_header = not self._header_written
#                 with open(self.path, 'a', newline='') as f:
#                     w = csv.DictWriter(f, fieldnames=hdr)
#                     if need_header:
#                         w.writeheader()
#                         self._header_written = True
#                     w.writerow(row)
#         csvlog = _CSVLogger(args.csv_path)
#         print(f"[CSV] logging to {args.csv_path}")

#     set_seed(args.seed)
#     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # --- build model ---
#     ModelCls=MODEL_MAP[args.model]

#     # text task kwargs for small models
#     extra_kwargs = {}
#     if args.task == 'textcls' and args.model in ('small_bert', 'small_transformer', 'distillbert_hf_gate'):
#         extra_kwargs['text_task'] = 'cls'
#         extra_kwargs['text_num_classes'] = args.num_labels

#     model = ModelCls(
#         insert_layers=args.insert_layers,
#         filter_backend=args.filter_backend, lp_k=args.lp_k,
#         gate_tau=args.gate_tau,
#         use_gate=args.gate_enable, use_resmix=args.resmix_enable,
#         pca_enable=args.pca_enable, pca_k=args.pca_k, pca_amin=args.pca_amin,
#         feature_mask=args.feature_mask,
#         ff_gamma=args.ff_gamma, ff_amin=args.ff_amin,
#         ff_apply_on=args.ff_apply_on, ff_use_drv_gate=args.ff_use_drv_gate,
#         **extra_kwargs
#     ).to(device)
#     collate=getattr(model,'collate_fn',None)
#     train_loader,val_loader=build_sst2_loaders(args.batch, collate)

#     # --- optim ---
#     opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
#                           lr=args.weight_lr, eps=args.adam_eps, weight_decay=args.adam_wd)

#     # --- DCA init & hooks ---
#     dca=None
#     if args.dca_enable and (DCA is not None):
#         if DCACfg is not None:
#             dca_cfg = DCACfg(enable=True)
#             dca = DCA(cfg=dca_cfg, num_classes=args.num_labels)
#         else:
#             try:
#                 dca = DCA(num_classes=args.num_labels)
#             except TypeError:
#                 try: dca = DCA()
#                 except Exception as e:
#                     print(f"[WARN] DCA construction failed: {e}"); dca=None
#         if dca is not None and hasattr(dca, "attach_hooks"):
#             if not getattr(dca, "_attached", False):
#                 dca.attach_hooks(model); dca._attached=True; print("[DCA] hooks attached")

#     dca_drv=None
#     if args.dca_derivative_enable and (DCADerivative is not None):
#         try: dca_drv = DCADerivative()
#         except Exception as e:
#             print(f"[WARN] DCADerivative() failed, disable derivative gate: {e}")
#             dca_drv=None

#     resmix_nudger=SoftResMixNudger(model) if args.resmix_enable and args.resmix_from_dca_lr>0 else None

#     pca_mgr=PCAOverlapManager(k=args.pca_k, ema_decay=0.99, device=str(device)) if args.pca_enable else None
#     if pca_mgr is not None and hasattr(model, 'attach_pca_manager'):
#         model.attach_pca_manager(pca_mgr)

#     guard=OverfitGuard(OverfitGuardConfig(patience=args.pca_patience,
#                                           kappa=args.dca_drv_kappa, clip=1.0))

#     # --- train loop ---
#     state={'step':0,'best_acc':-1}
#     for epoch in range(1,args.epochs+1):
#         m=run_epoch(model,train_loader,opt,device,state,args,train=True)
#         print(f"[E{epoch:02d}] train loss={m['loss']:.4f} acc={m['acc']:.4f}")
#         if swan is not None:
#             swan.log({"train/epoch_loss": float(m['loss']), "train/epoch_acc": float(m['acc'])}, step=epoch)
#         if csvlog is not None:
#             csvlog.log({"stage":"train_epoch","epoch":epoch,
#                         "loss":round(float(m['loss']),6),"acc":round(float(m['acc']),6)})

#         if args.quickval_every>0 and state['step']%args.quickval_every==0:
#             opt.zero_grad(set_to_none=True)
#             quickval_arch_step(model,val_loader,device,args,dca,dca_drv,resmix_nudger,pca_mgr,guard)

#         vm=evaluate(model,val_loader,device,args)
#         print(f"[E{epoch:02d}] valid loss={vm['loss']:.4f} acc={vm['acc']:.4f}")
#         if swan is not None:
#             swan.log({"valid/epoch_loss": float(vm['loss']), "valid/epoch_acc": float(vm['acc'])}, step=epoch)
#         if csvlog is not None:
#             csvlog.log({"stage":"valid_epoch","epoch":epoch,
#                         "loss":round(float(vm['loss']),6),"acc":round(float(vm['acc']),6)})

#         if vm['acc']>state['best_acc']:
#             state['best_acc']=vm['acc']
#             os.makedirs(args.save_dir,exist_ok=True)
#             torch.save({'model':model.state_dict(),'args':vars(args),'epoch':epoch},
#                        os.path.join(args.save_dir,args.save_name))
#             print(f"[E{epoch:02d}] saved best to {os.path.join(args.save_dir,args.save_name)}")

# if __name__=='__main__':
# #     main()

# '''

# python optimization/train_eval_final.py \
#   --task textcls --text_dataset sst2 --model small_bert \
#   --epochs 40 --batch 64 --weight_lr 3e-4 \
#   --insert_layers last \
#   --gate_enable --resmix_enable --feature_mask \
#   --filter_backend lp_fixed --lp_k 3 \
#   --dca_enable --dca_mode e_ctr_resmix --dca_beta 1.0 \
#   --dca_derivative_enable --dca_drv_mode ema --dca_drv_lambda 0.05 --dca_drv_kappa 3.0 \
#   --resmix_from_dca_lr 1e-4 \
#   --log_dir optimization/log/textcls/sst2/smallbert/fft_A \
#   --run_tag sst2_smallbert_fft_A \
#   --save_dir optimization/checkpoints/sst2/smallbert/fft \
#   --save_name sst2_smallbert_fft_A.pt \
#   --log_every 50 --quickval_every 600 --arch_val_batches 1 \
#   --swan_enable --swan_project dca_suite \
#   --swan_dir optimization/log/swan \
#   --swan_experiment sst2_smallbert_fft_A \
#   --csv_log --csv_path optimization/log/textcls/sst2/smallbert/fft_A/metrics.csv
# '''


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Final train_eval.py (PCA/FFT denoise + OverfitGuard + robust text batch handling).
# - Robust batch parsing: dict / (x,y) / (x,mask,y) / ((x,mask),y) / raw text
# - Auto-tokenize raw text to BERT ids when needed (bert-base-uncased)
# - Strictly denoise-only (no train→valid compensation)
# - SwanLab logging + CSV logging
# """

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Final train_eval.py (text: SST-2 | vision: MNIST/CIFAR10/CIFAR100)
# - Robust batch parsing: dict / (x,y) / (x,mask,y) / ((x,mask),y) / raw text
# - Auto-tokenize raw text to BERT ids when needed (bert-base-uncased)
# - Strict denoise-only; supports PCA soft/hard, DCA/derivative, ResMix (energy-guided)
# - SwanLab logging + CSV logging
# """

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Unified train_eval_final.py
# - Text (SST-2) and Vision (MNIST / CIFAR-10 / CIFAR-100)
# - PCA/FFT denoise + OverfitGuard + energy-guided nudge
# - SwanLab logging + CSV logging
# - Strict base switch to disable all inserts
# """
# import warnings as _warnings
# _warnings.filterwarnings(
#     "ignore",
#     message=r"The pynvml package is deprecated.*",
#     category=FutureWarning
# )
# import os, argparse, random, sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from datasets import load_dataset  # for text datasets (SST-2)

# # ---------------------------
# # Globals for logging backends
# # ---------------------------
# swan = None     # SwanLab run handle
# csvlog = None   # CSV logger handle

# # --- optional: transformers tokenizer for raw text ---
# _TOKENIZER = None
# def _get_tokenizer():
#     global _TOKENIZER
#     if _TOKENIZER is None:
#         try:
#             from transformers import AutoTokenizer
#             _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
#         except Exception as e:
#             raise RuntimeError(
#                 f"Need transformers to tokenize raw text for small_bert. Install it or "
#                 f"provide a model.collate_fn. Error: {e}"
#             )
#     return _TOKENIZER

# # --- DCA & friends (robust imports) ---
# try:
#     from modules.dca import DCACfg, DCA
# except Exception as e:
#     DCACfg = None; DCA = None
#     print(f"[WARN] DCA unavailable: {e}")

# try:
#     from modules.dca_derivative import DCADerivative
# except Exception as e:
#     DCADerivative = None
#     print(f"[WARN] DCADerivative unavailable: {e}")

# try:
#     from modules.dca_e_ctr_resmix import build_e_ctr_resmix
# except Exception as e:
#     build_e_ctr_resmix = None
#     print(f"[WARN] build_e_ctr_resmix unavailable: {e}")

# try:
#     from modules.spec_reg import spectral_penalty_depthwise
# except Exception:
#     spectral_penalty_depthwise = None

# from modules.soft_resmix import SoftResMixNudger
# from modules.pca_overlap import PCAOverlapManager
# from modules.overfit_guard import OverfitGuard, OverfitGuardConfig

# # --- models ---
# from models.distillbert_hf_gate import DistillBertHFGate
# from models.small_bert import SmallBERT
# from models.small_transformer import SmallTransformer

# # vision dataloaders
# try:
#     from torchvision import datasets as tvds, transforms as T
# except Exception:
#     tvds = None; T = None

# # 可选：baseline（无插桩）对比
# # from models.distillbert_hf import DistilBertHF

# MODEL_MAP = {
#     'distillbert_hf_gate': DistillBertHFGate,
#     'small_bert': SmallBERT,
#     'small_transformer': SmallTransformer,
#     # 'distillbert_hf': DistilBertHF,  # 可选：baseline
# }

# def set_seed(seed=42):
#     random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# # ---------------------------
# # Data: text (SST-2)
# # ---------------------------
# # def build_sst2_loaders(batch, collate_fn=None):
# #     ds = load_dataset("glue", "sst2")
# #     train = list(zip(ds['train']['sentence'], ds['train']['label']))
# #     val   = list(zip(ds['validation']['sentence'], ds['validation']['label']))
# #     return DataLoader(train, batch_size=batch, shuffle=True, collate_fn=collate_fn, drop_last=True), \
# #            DataLoader(val,   batch_size=batch, shuffle=False, collate_fn=collate_fn)


# def three_way_split_indices(n, train_pool_frac=0.8, inner_train_frac=0.9, seed=42):
#     g = torch.Generator().manual_seed(int(seed))
#     idx = torch.randperm(n, generator=g).tolist()
#     n_train_pool = int(round(n * train_pool_frac))
#     pool = idx[:n_train_pool]
#     test_idx = idx[n_train_pool:]             # 20%

#     n_train = int(round(len(pool) * inner_train_frac))
#     train_idx = pool[:n_train]                # 0.8*0.9 = 72%
#     val_idx   = pool[n_train:]                # 0.8*0.1 = 8%
#     return train_idx, val_idx, test_idx

# # --- 原 build_sst2_loaders 改成返回 (train, val, test=None) ---
# # def build_sst2_loaders(batch, collate_fn=None):
# #     ds = load_dataset("glue", "sst2")
# #     train = list(zip(ds['train']['sentence'], ds['train']['label']))
# #     val   = list(zip(ds['validation']['sentence'], ds['validation']['label']))
# #     train_loader = DataLoader(train, batch_size=batch, shuffle=True,  collate_fn=collate_fn, drop_last=True)
# #     val_loader   = DataLoader(val,   batch_size=batch, shuffle=False, collate_fn=collate_fn)
# #     test_loader  = None  # GLUE/SST-2 的 test 没有公开 label，跳过
# #     return train_loader, val_loader, test_loader

# def build_sst2_loaders(batch, collate_fn=None, val_split=0.1, seed=42):
#     ds = load_dataset("glue", "sst2")
#     train_full = list(zip(ds['train']['sentence'], ds['train']['label']))
#     # 官方 validation 直接作为 test（符合你要求）
#     test_full  = list(zip(ds['validation']['sentence'], ds['validation']['label']))

#     # 从 train_full 里切出 valid
#     n_total = len(train_full)
#     n_val   = max(1, int(round(n_total * float(val_split))))
#     g = torch.Generator().manual_seed(int(seed))
#     perm = torch.randperm(n_total, generator=g).tolist()
#     val_idx   = set(perm[:n_val])
#     train_set = [train_full[i] for i in range(n_total) if i not in val_idx]
#     val_set   = [train_full[i] for i in perm[:n_val]]

#     train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
#                               collate_fn=collate_fn, drop_last=True)
#     val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False,
#                               collate_fn=collate_fn)
#     test_loader  = DataLoader(test_full, batch_size=batch, shuffle=False,
#                               collate_fn=collate_fn)
#     return train_loader, val_loader, test_loader


# def _iter_pca_wrappers(model):
#     """
#     统一遍历：支持 dict / list / tuple / 自定义属性名
#     """
#     for name in ('pca_wrappers', 'pca_modules'):
#         w = getattr(model, name, None)
#         if isinstance(w, dict):
#             for lid, wrap in w.items():
#                 yield lid, wrap
#             return
#         if isinstance(w, (list, tuple)):
#             for lid, wrap in enumerate(w):
#                 yield lid, wrap
#             return
#     return  # 没有 PCA wrapper

# def _log_pca_apply(tag, lid, O_i, O_res, g_before, g_after, prob=None, mask=None, a_min=None):
#     """
#     打印一行可读日志，涵盖：平均重叠、g 的变化、硬采样概率、选中的分量索引等。
#     - O_i, O_res, g_before, g_after: Tensor 或 None
#     - prob: 采样概率向量（Bernoulli 时）
#     - mask: 0/1 的选择掩码（硬采样时）
#     """
#     def _stats(x):
#         if x is None: return "n/a"
#         x = x.float().view(-1)
#         return f"min/mean/max={x.min():.2f}/{x.mean():.2f}/{x.max():.2f}"

#     msg = [f"[{tag}][PCA] lid={lid}"]

#     if O_i is not None:
#         msg.append(f"O_i(mean)={float(O_i.mean().item()):.3f}")
#     if O_res is not None:
#         msg.append(f"O_res(mean)={float(O_res.mean().item()):.3f}")

#     if g_before is not None:
#         msg.append(f"g_before({_stats(g_before)})")
#     if g_after is not None:
#         msg.append(f"g_after({_stats(g_after)})")

#     if prob is not None:
#         msg.append(f"p_comp({_stats(prob)})")
#     if mask is not None:
#         sel = (mask > 0.5).nonzero(as_tuple=False).flatten()
#         msg.append(f"selected={int(sel.numel())}/{int(mask.numel())}")
#         # 只打前 12 个索引，避免刷屏
#         if sel.numel() > 0:
#             head = ", ".join([str(int(i)) for i in sel[:12]])
#             if sel.numel() > 12: head += " ..."
#             msg.append(f"indices=[{head}]")
#     if a_min is not None:
#         msg.append(f"a_min={a_min:.2f}")

#     print(" ".join(msg))

# # def build_vision_loaders(name: str, batch: int):
# #     assert tvds is not None and T is not None, "Need torchvision for vision datasets"
# #     if name == 'mnist':
# #         tr = T.Compose([T.ToTensor()])
# #         te = T.Compose([T.ToTensor()])
# #         train = tvds.MNIST('data', train=True, download=True, transform=tr)
# #         val   = tvds.MNIST('data', train=False, download=True, transform=te)
# #         num_labels = 10
# #     elif name == 'c10':
# #         tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
# #                         T.ToTensor()])
# #         te = T.Compose([T.ToTensor()])
# #         train = tvds.CIFAR10('data', train=True, download=True, transform=tr)
# #         val   = tvds.CIFAR10('data', train=False, download=True, transform=te)
# #         num_labels = 10
# #     elif name == 'c100':
# #         tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
# #                         T.ToTensor()])
# #         te = T.Compose([T.ToTensor()])
# #         train = tvds.CIFAR100('data', train=True, download=True, transform=tr)
# #         val   = tvds.CIFAR100('data', train=False, download=True, transform=te)
# #         num_labels = 100
# #     else:
# #         raise ValueError(f"Unknown vision dataset: {name}")
# #     return DataLoader(train, batch_size=batch, shuffle=True, drop_last=True), \
# #            DataLoader(val,   batch_size=batch, shuffle=False), num_labels
# # --- 原 build_vision_loaders 改成返回 (train, val, test, num_labels)
# def build_vision_loaders(name: str, batch: int, root: str="data",
#                          val_split: float=0.1, seed: int=42):
#     assert tvds is not None and T is not None, "Need torchvision for vision datasets"

#     if name == 'mnist':
#         tr = T.Compose([T.ToTensor()])
#         te = T.Compose([T.ToTensor()])
#         train_full = tvds.MNIST(root, train=True,  download=True, transform=tr)
#         test       = tvds.MNIST(root, train=False, download=True, transform=te)
#         num_labels = 10

#     elif name == 'c10':
#         tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
#         te = T.Compose([T.ToTensor()])
#         train_full = tvds.CIFAR10(root, train=True,  download=True, transform=tr)
#         test       = tvds.CIFAR10(root, train=False, download=True, transform=te)
#         num_labels = 10

#     elif name == 'c100':
#         tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
#         te = T.Compose([T.ToTensor()])
#         train_full = tvds.CIFAR100(root, train=True,  download=True, transform=tr)
#         test       = tvds.CIFAR100(root, train=False, download=True, transform=te)
#         num_labels = 100

#     else:
#         raise ValueError(f"Unknown vision dataset: {name}")

#     # 从训练集里切 valid
#     n_total = len(train_full)
#     n_val   = max(1, int(round(n_total * float(val_split))))
#     g = torch.Generator().manual_seed(int(seed))
#     perm = torch.randperm(n_total, generator=g).tolist()
#     val_idx = set(perm[:n_val])
#     from torch.utils.data import Subset
#     train_ds = Subset(train_full, [i for i in range(n_total) if i not in val_idx])
#     val_ds   = Subset(train_full, [i for i in perm[:n_val]])

#     train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  drop_last=True)
#     val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
#     test_loader  = DataLoader(test,     batch_size=batch, shuffle=False)
#     return train_loader, val_loader, test_loader, num_labels


# # ---------------------------
# # Utils
# # ---------------------------
# def accuracy(logits, labels):
#     preds = logits.argmax(-1)
#     return (preds == labels).float().mean().item()
# # Robust batch parser + auto-tokenize
# def _maybe_tokenize_text(x_like, max_len, device):
#     # list/tuple of strings
#     if isinstance(x_like, (list, tuple)) and len(x_like) > 0 and isinstance(x_like[0], str):
#         tok = _get_tokenizer()
#         enc = tok(
#             list(x_like),
#             padding='max_length',
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt"
#         )
#         return enc["input_ids"].to(device), enc["attention_mask"].to(device)
#     # single string
#     if isinstance(x_like, str):
#         tok = _get_tokenizer()
#         enc = tok(
#             [x_like],
#             padding='max_length',
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt"
#         )
#         return enc["input_ids"].to(device), enc["attention_mask"].to(device)
#     return None, None

# def parse_batch(batch, device, max_len=128):
#     """
#     Normalize batch to (x_ids, attn_mask, labels).
#     Supports:
#       - dict: {'input_ids','attention_mask','labels'} or {'x':(ids,mask),'labels':y}
#       - (x, y), (x, mask, y), ((x, mask), y), ({'input_ids':..., 'attention_mask':...}, y)
#       - raw text: list[str] or str  -> auto tokenize to ids/mask
#       - vision: (image_tensor, label)
#     """
#     def to_dev(t):
#         return t.to(device) if torch.is_tensor(t) else t

#     def norm_x(xlike):
#         # dict path
#         if isinstance(xlike, dict):
#             xi = xlike.get('input_ids', xlike.get('x', xlike.get('inputs')))
#             xm = xlike.get('attention_mask', xlike.get('mask'))
#             ids, am = _maybe_tokenize_text(xi, max_len, device)
#             if ids is not None:
#                 return ids, am
#             if (xm is None) and isinstance(xi, (tuple, list)) and len(xi) == 2:
#                 xi, xm = xi
#             return to_dev(xi), to_dev(xm)
#         # tuple/list path
#         if isinstance(xlike, (tuple, list)):
#             if len(xlike) == 2 and torch.is_tensor(xlike[0]):
#                 return to_dev(xlike[0]), to_dev(xlike[1])
#             ids, am = _maybe_tokenize_text(xlike, max_len, device)
#             if ids is not None:
#                 return ids, am
#             if len(xlike) >= 1:
#                 return to_dev(xlike[0]), None
#         # single tensor -> ids only
#         if torch.is_tensor(xlike):
#             return to_dev(xlike), None
#         # raw text string
#         ids, am = _maybe_tokenize_text(xlike, max_len, device)
#         if ids is not None:
#             return ids, am
#         # unknown
#         return xlike, None

#     if isinstance(batch, dict):
#         xi, xm = norm_x(batch)
#         y = batch.get('labels', batch.get('y', None))
#         return xi, xm, to_dev(y)

#     if isinstance(batch, (tuple, list)):
#         if len(batch) == 2:
#             a, b = batch
#             xi, xm = norm_x(a)
#             y = b
#             return xi, xm, to_dev(y)
#         if len(batch) == 3:
#             xi, xm = norm_x(batch[0])
#             xm = batch[1] if (xm is None) else xm
#             y = batch[2]
#             return xi, to_dev(xm), to_dev(y)
#         xi, xm = norm_x(batch[0])
#         if xm is None and len(batch) > 2:
#             xm = batch[1]
#         y = batch[-1]
#         return xi, to_dev(xm), to_dev(y)

#     if torch.is_tensor(batch):
#         return batch.to(device), None, None

#     ids, am = _maybe_tokenize_text(batch, max_len, device)
#     if ids is not None:
#         return ids, am, None

#     raise TypeError(f"Unsupported batch type: {type(batch)}")

# def call_model(model, x_ids, attn_mask):
#     """
#     Unified call:
#     - DistilBert-like expects attention_mask keyword
#     - SmallBERT/SmallTransformer: forward(x, attn_mask=...) or forward(x)
#     """
#     if hasattr(model, 'bert') or model.__class__.__name__.lower().startswith('distill'):
#         return model(x_ids, attention_mask=attn_mask)
#     try:
#         return model(x_ids, attn_mask=attn_mask) if attn_mask is not None else model(x_ids)
#     except TypeError:
#         return model(x_ids)

# # ---------------------------
# # Train / Eval
# # ---------------------------
# # def run_epoch(model, loader, opt, device, state, args, train=True):
# def run_epoch(model, loader, opt, device, state, args, train=True, pca_mgr=None):

#     model.train(train)
#     tot_loss=tot_acc=n=0
#     for batch in loader:
#         if hasattr(model, 'collate_fn'):
#             batch = model.collate_fn(batch)
#         x_ids, attn_mask, labels = parse_batch(batch, device, max_len=args.max_len)
#         assert torch.is_tensor(x_ids), f"x_ids must be Tensor, got {type(x_ids)}"

#         if train: opt.zero_grad(set_to_none=True)
#         out = call_model(model, x_ids, attn_mask)
#         logits = out['logits']
#         assert logits.dim() == 2 and logits.size(1) == args.num_labels, \
#             f"expect logits [B,{args.num_labels}], got {tuple(logits.shape)}"

#         if labels is None:
#             continue
#         ce_now = out.get('loss') or F.cross_entropy(logits, labels)
#         if train:
#             with torch.no_grad():
#                 model._last_train_ce = ce_now.detach()
#         loss = ce_now

#         # --- 训练侧：持续更新 PCA 的 train 统计 ---
#         if train and (getattr(args, 'pca_enable', False)) and (hasattr(model, 'pca_wrappers')) and (pca_mgr is not None):
#             for lid, wrap in getattr(model, 'pca_wrappers', {}).items():
#                 Xtr = getattr(wrap, '_last_input', None)
#                 if Xtr is None:
#                     continue
#                 if Xtr.dim() == 3:      # [B, T, D] -> [N, D]
#                     Xt = Xtr.reshape(-1, Xtr.shape[-1])
#                 elif Xtr.dim() == 2:    # [N, D]
#                     Xt = Xtr
#                 else:
#                     continue
#                 # 累积训练均值/协方差（内部做 EMA）
#                 try:
#                     pca_mgr.update_train(lid, Xt)
#                 except Exception as e:
#                     # 不让训练中断；有问题打印警告即可
#                     print(f"[PCA][update_train] lid={lid} warn: {e}")


#         acc = accuracy(logits, labels)
#         if train:
#             loss.backward(); opt.step(); state['step']+=1
#         bs=labels.size(0); tot_loss+=loss.item()*bs; tot_acc+=acc*bs; n+=bs

#         if train and args.log_every>0 and state['step']%args.log_every==0:
#             mean_loss=tot_loss/max(1,n); mean_acc=tot_acc/max(1,n)
#             print(f"[S{state['step']}] train loss={mean_loss:.4f} acc={mean_acc:.4f}")
#             try:
#                 if swan is not None:
#                     swan.log({"train/step_loss": float(mean_loss),
#                               "train/step_acc": float(mean_acc)}, step=state['step'])
#                 if csvlog is not None:
#                     csvlog.log({"stage":"train_step","step":state['step'],"epoch":-1,
#                                 "loss":round(float(mean_loss),6),"acc":round(float(mean_acc),6)})
#             except Exception:
#                 pass

#     return {'loss':tot_loss/max(1,n),'acc':tot_acc/max(1,n)}

# @torch.no_grad()
# def evaluate(model, loader, device, args):
#     model.eval(); tot_loss=tot_acc=n=0
#     for batch in loader:
#         if hasattr(model, 'collate_fn'):
#             batch = model.collate_fn(batch)
#         x_ids, attn_mask, labels = parse_batch(batch, device, max_len=args.max_len)
#         assert torch.is_tensor(x_ids), f"x_ids must be Tensor, got {type(x_ids)}"
#         out = call_model(model, x_ids, attn_mask)
#         logits = out['logits']
#         assert logits.dim() == 2 and logits.size(1) == args.num_labels, \
#             f"expect logits [B,{args.num_labels}], got {tuple(logits.shape)}"

#         loss = out.get('loss') or F.cross_entropy(logits, labels)
#         acc = accuracy(logits, labels)
#         bs=labels.size(0); tot_loss+=loss.item()*bs; tot_acc+=acc*bs; n+=bs
#     return {'loss':tot_loss/max(1,n),'acc':tot_acc/max(1,n)}


# # ---------- NEW/REPLACE: quickval_arch_step ----------
# def quickval_arch_step(model, val_loader, device, args, dca, dca_drv, resmix_nudger, pca_mgr, guard, qv_state):
#     model.eval()
#     import itertools
#     val_iter = itertools.islice(iter(val_loader), int(max(1, args.arch_val_batches)))

#     def _stat3(t: torch.Tensor):
#         t = t.float().view(-1)
#         return f"{t.min().item():.4f}/{t.mean().item():.4f}/{t.max().item():.4f}"

#     ce_list, acc_list = [], []

#     # 1) 小批验证代理
#     with torch.no_grad():
#         for batch in val_iter:
#             if isinstance(batch, (list, tuple)):
#                 xb, yb = batch[0].to(device), batch[1].to(device)
#                 fwd_kwargs = {}
#             elif isinstance(batch, dict):
#                 yb = batch.get('labels', batch.get('label'))
#                 if yb is None:
#                     raise RuntimeError("quickval batch dict has no 'labels' key.")
#                 fwd_kwargs = {k: (v.to(device) if torch.is_tensor(v) else v)
#                               for k, v in batch.items() if k not in ['labels', 'label']}
#                 xb = None
#                 yb = yb.to(device)
#             else:
#                 raise RuntimeError(f"Unsupported batch type in quickval: {type(batch)}")

#             out = model(xb) if (xb is not None and len(fwd_kwargs) == 0) else model(**fwd_kwargs)
#             logits = out if isinstance(out, torch.Tensor) else out['logits'] if isinstance(out, dict) else out.logits
#             ce = F.cross_entropy(logits, yb).item()
#             acc = (logits.argmax(dim=-1) == yb).float().mean().item()
#             ce_list.append(ce); acc_list.append(acc)

#     num = max(1, len(ce_list))
#     val_ce  = float(sum(ce_list) / num)
#     val_acc = float(sum(acc_list) / num)

#     # 2) 训练侧 proxy + 导数权重
#     train_ce_proxy = qv_state['train_ce_ema'].v if qv_state['train_ce_ema'].v is not None else val_ce
#     drv_weight = 0.0
#     if dca_drv is not None:
#         try: drv_weight = float(dca_drv.weight())
#         except Exception: drv_weight = 0.0

#     # # 3) OverfitGuard
#     # armed, guard_w, extras = guard.tick(train_proxy=train_ce_proxy, val_proxy=val_ce)
#     # if not ((qv_state['train_acc_ema'].v is not None) and (qv_state['train_acc_ema'].v >= float(args.trigger_train_acc))):
#     #     armed = False; guard_w = 0.0
#     # if getattr(args, 'pca_force', False):
#     #     armed = True

#     # -------------------------
# # 3) OverfitGuard 判定（分离 train/val proxy）+ 训练精度门控
#     # -------------------------
#     armed, guard_w, extras = guard.tick(train_proxy=train_ce_proxy, val_proxy=val_ce)

#     # 训练精度门控：只有当 train_acc 的 EMA ≥ 阈值时才允许触发（除非 pca_force）
#     train_acc_ema = qv_state['train_acc_ema'].v
#     train_acc_gate_ok = (train_acc_ema is not None) and (train_acc_ema >= float(args.trigger_train_acc))

#     if not getattr(args, 'pca_force', False) and not train_acc_gate_ok:
#         armed = False
#         guard_w = 0.0

#     # 调试信息（可选）
#     print(f"[QVAL] guard_armed={armed} guard_w={guard_w:.3f} "
#         f"train_acc_ema={None if train_acc_ema is None else round(train_acc_ema,4)} "
#         f"gate_ok={train_acc_gate_ok} drv_w={drv_weight:.3f} "
#         f"val_ce={val_ce:.4f} val_acc={val_acc:.4f}")

        
#     # ---------- Trigger mode override ----------
#     mode = getattr(args, 'trigger_mode', 'worsen')
#     armed_by_mode = False
#     mode_weight = 0.0  # 给不同模式一个保守的强度（和 guard_w 取 max）

#     if not getattr(args, 'pca_force', False) and train_acc_gate_ok:
#         if mode == 'worsen':
#             # 保持 OverfitGuard 结果
#             armed_by_mode = armed
#             mode_weight = guard_w

#         elif mode == 'stagnate':
#             # 更新 best_val_acc 与停滞计数
#             if (qv_state['best_val_acc'] is None) or \
#             (val_acc >= qv_state['best_val_acc'] + float(args.stagnation_tol)):
#                 qv_state['best_val_acc'] = val_acc
#                 qv_state['stg_streak'] = 0
#             else:
#                 qv_state['stg_streak'] = qv_state.get('stg_streak', 0) + 1

#             patience = int(args.stagnation_patience or args.drift_patience)
#             if qv_state['stg_streak'] >= patience:
#                 armed_by_mode = True
#                 # 给一个适中的强度；你也可以根据停滞长度线性放大
#                 mode_weight = 0.5

#         elif mode == 'gap':
#             tr_acc = float(qv_state['train_acc_ema'].v or 0.0)
#             gap = max(0.0, tr_acc - float(val_acc))
#             if gap >= float(args.gap_thr):
#                 armed_by_mode = True
#                 # 根据 gap 线性映射一个强度(0~1)
#                 mode_weight = min(1.0, max(0.0, (gap - args.gap_thr) / max(1e-6, 1.0 - args.gap_thr)))
#         else:
#             # 未知模式，退回原逻辑
#             armed_by_mode = armed
#             mode_weight = guard_w

#     # 合并结果（force 优先）
#     if getattr(args, 'pca_force', False):
#         armed = True
#     else:
#         armed = armed_by_mode
#         guard_w = max(float(guard_w), float(mode_weight))

#     # 可选：打印当前模式与权重，方便观测
#     print(f"[TRIGGER] mode={mode} armed={armed} guard_w={guard_w:.3f} "
#         f"train_acc_ema={(qv_state['train_acc_ema'].v or 0.0):.4f} val_acc={val_acc:.4f}")
#     # -------------------------------------------


#     # 4) 找 PCA wrappers
#     # pca_wrappers = []
#     # if hasattr(model, 'pca_wrappers') and isinstance(model.pca_wrappers, (list, tuple)):
#     #     pca_wrappers = list(model.pca_wrappers)
#     # elif hasattr(model, 'pca_modules') and isinstance(model.pca_modules, (list, tuple)):
#     #     pca_wrappers = list(model.pca_modules)

#     # if len(pca_wrappers) == 0 or (pca_mgr is None):
#     #     qv_state['val_acc_last'] = val_acc
#     #     return

#     # 4) 找 PCA wrappers（兼容 dict / list / tuple）
#     pca_wrappers = []
#     if hasattr(model, 'pca_wrappers'):
#         w = model.pca_wrappers
#         if isinstance(w, dict):
#             pca_wrappers = [w[k] for k in sorted(w.keys())]  # 用层号排序
#         elif isinstance(w, (list, tuple)):
#             pca_wrappers = list(w)
#     elif hasattr(model, 'pca_modules') and isinstance(model.pca_modules, (list, tuple)):
#         pca_wrappers = list(model.pca_modules)

#     if len(pca_wrappers) == 0 or (pca_mgr is None):
#         qv_state['val_acc_last'] = val_acc
#         return


#     # 5) 触发：应用 soft/hard，并保存 snapshot
#     if armed:
#         qv_state.setdefault('g_snap', {})  # 用于 rollback_to=snapshot
#         # for l, wrapper in enumerate(pca_wrappers):
#         #     # 取 rho
#         #     rho = None
#         #     if hasattr(pca_mgr, 'compute_overlap'):
#         #         rho = pca_mgr.compute_overlap(layer=l)
#         #     elif hasattr(pca_mgr, 'get_overlap'):
#         #         rho = pca_mgr.get_overlap(l)
#         #     # if rho is None:
#         #     #     if hasattr(wrapper, 'get_component_gains') and (wrapper.get_component_gains() is not None):
#         #     #         kshape = wrapper.get_component_gains().shape
#         #     #         rho = torch.ones(kshape, device=device, dtype=torch.float32)
#         #     #     else:
#         #     #         rho = torch.tensor(1.0, device=device, dtype=torch.float32)
#         #     # if not torch.is_tensor(rho):
#         #     #     rho = torch.tensor([float(rho)], device=device, dtype=torch.float32)
#         #     if rho is None:
#         #         k = int(getattr(wrapper, 'k', 1))
#         #         rho = torch.ones(k, device=device, dtype=torch.float32)
#         #     if not torch.is_tensor(rho):
#         #         rho = torch.tensor([float(rho)], device=device, dtype=torch.float32)
#         #     if rho.dim() == 0:
#         #         rho = rho.view(1)

#         #     rho = rho.float().to(device)

#         for l, wrapper in enumerate(pca_wrappers):
#             # --- 用验证激活算 overlap -> rho ---
#             Xval = getattr(wrapper, '_last_input', None)
#             if Xval is None:
#                 # 该层这次 quickval 没跑到，跳过
#                 print(f"[PCA] lid={l} skip: no _last_input on wrapper")
#                 continue

#             # 统一成 [N, C]（文本常见 [B,T,D]；若本来是 [N,C] 直接用）
#             if Xval.dim() == 3:          # [B, T, D]
#                 Xv = Xval.reshape(-1, Xval.shape[-1])
#             elif Xval.dim() == 2:        # [N, C]
#                 Xv = Xval
#             else:
#                 print(f"[PCA] lid={l} skip: unsupported X shape {tuple(Xval.shape)}")
#                 continue

#             # 计算每个主成分与训练侧的 overlap 分数
#             scores = pca_mgr.overlap_scores(l, Xv)  # -> {'O_i': [k], 'O_res': [1] or [k']}
#             rho = scores['O_i'].float().to(device)  # 只用 component 侧的 overlap 当 rho

#             # 读 g_before & 存快照
#             if hasattr(wrapper, 'get_component_gains') and (wrapper.get_component_gains() is not None):
#                 g_before = wrapper.get_component_gains().detach().clone()
#             elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
#                 g_before = wrapper.comp_gain.detach().clone()
#             else:
#                 g_before = None
#             if g_before is not None:
#                 qv_state['g_snap'][l] = g_before.clone()

#             strength = float(max(guard_w, drv_weight))
#             a_min = float(getattr(args, 'pca_amin', 0.85))

#             if getattr(args, 'arm_hard', False):
#                 # HARD：概率 & 掩码
#                 scale   = float(getattr(args, 'arm_prob_scale', 1.0))
#                 minprob = float(getattr(args, 'arm_minprob', 0.05))
#                 maxprob = float(getattr(args, 'arm_maxprob', 0.95))
#                 p_comp  = torch.clamp(scale * (1.0 - rho), min=minprob, max=maxprob)

#                 mode = getattr(args, 'arm_hard_mode', 'bernoulli')
#                 if mode == 'bernoulli':
#                     m_comp = torch.bernoulli(p_comp).to(rho.dtype)
#                 else:
#                     score = (1.0 - rho)
#                     K_cfg = int(getattr(args, 'arm_hard_topk', 0))
#                     if K_cfg > 0:
#                         K = min(K_cfg, score.numel())
#                     else:
#                         K = int(torch.clamp(p_comp.sum().round(), min=1.0).item())
#                         K = min(K, score.numel())
#                     top_idx = torch.topk(score, k=K, largest=True).indices
#                     m_comp = torch.zeros_like(score); m_comp[top_idx] = 1.0

#                 g_hard = (1.0 - m_comp) + m_comp * a_min
#                 adj_g  = 1.0 - strength * (1.0 - g_hard)

#                 if hasattr(wrapper, 'set_component_gains'):
#                     wrapper.set_component_gains(adj_g)
#                 elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
#                     with torch.no_grad():
#                         wrapper.comp_gain.copy_(adj_g.to(wrapper.comp_gain.device, dtype=wrapper.comp_gain.dtype))
#                 if hasattr(wrapper, 'set_res_gain'): wrapper.set_res_gain(1.0)
#                 if hasattr(wrapper, 'set_blend'):    wrapper.set_blend(1.0)

#                 # 打印：概率/选中/步幅
#                 sel = (m_comp > 0.5).nonzero(as_tuple=False).flatten()
#                 head = ", ".join([str(int(i)) for i in sel[:12].tolist()]) + (" ..." if sel.numel() > 12 else "")
#                 if g_before is not None:
#                     g_after = adj_g if isinstance(adj_g, torch.Tensor) else torch.as_tensor(adj_g, device=g_before.device, dtype=g_before.dtype)
#                     d = (g_after - g_before).abs()
#                     print(f"[ARM-HARD] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
#                           f"p_comp(min/mean/max)={_stat3(p_comp)} selected={int(sel.numel())}/{m_comp.numel()} "
#                           f"indices=[{head}] |Δg|(min/mean/max)={_stat3(d)}")
#                 else:
#                     print(f"[ARM-HARD] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
#                           f"p_comp(min/mean/max)={_stat3(p_comp)} selected={int(sel.numel())}/{m_comp.numel()} indices=[{head}]")

#             else:
#                 # SOFT：base_g -> adj_g
#                 base_g = (1.0 - args.pca_lambda * (1.0 - rho)).clamp(min=a_min, max=1.0)
#                 adj_g  = 1.0 - strength * (1.0 - base_g)

#                 if hasattr(wrapper, 'set_component_gains'):
#                     wrapper.set_component_gains(adj_g)
#                 elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
#                     with torch.no_grad():
#                         wrapper.comp_gain.copy_(adj_g.to(wrapper.comp_gain.device, dtype=wrapper.comp_gain.dtype))
#                 if hasattr(wrapper, 'set_res_gain'): wrapper.set_res_gain(1.0)
#                 if hasattr(wrapper, 'set_blend'):    wrapper.set_blend(1.0)

#                 if g_before is not None:
#                     g_after = adj_g if isinstance(adj_g, torch.Tensor) else torch.as_tensor(adj_g, device=g_before.device, dtype=g_before.dtype)
#                     d = (g_after - g_before).abs()
#                     print(f"[ARM-SOFT] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
#                           f"rho(min/mean/max)={_stat3(rho)} |Δg|(min/mean/max)={_stat3(d)}")
#                 else:
#                     print(f"[ARM-SOFT] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
#                           f"rho(min/mean/max)={_stat3(rho)}")

#         if resmix_nudger is not None:
#             try: resmix_nudger.nudge(step_size=-0.05 * float(max(guard_w, drv_weight)))
#             except Exception: pass

#         qv_state['armed_live'] = True
#         qv_state['val_acc0']   = val_acc
#         qv_state['rb_streak']  = 0

#     # 6) 回退判定
#     if qv_state['val_acc_last'] is None:
#         qv_state['val_acc_last'] = val_acc

#     if qv_state['armed_live']:
#         improved = (val_acc >= qv_state['val_acc0'] - float(args.stagnation_tol))
#         qv_state['rb_streak'] = 0 if improved else (qv_state['rb_streak'] + 1)

#         if qv_state['rb_streak'] >= int(args.rollback_patience):
#             gamma = float(getattr(args, 'rollback_gamma', 0.6))
#             to_snapshot = (getattr(args, 'rollback_to', 'one') == 'snapshot')
#             g_snap = qv_state.get('g_snap', {})

#             for lid, wrapper in enumerate(pca_wrappers):
#                 # 当前 g
#                 if hasattr(wrapper, 'get_component_gains') and (wrapper.get_component_gains() is not None):
#                     g_now = wrapper.get_component_gains().detach()
#                 elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
#                     g_now = wrapper.comp_gain.detach()
#                 else:
#                     continue

#                 # 目标
#                 if to_snapshot and (lid in g_snap):
#                     tgt = g_snap[lid].to(g_now.device, dtype=g_now.dtype)
#                 else:
#                     tgt = torch.ones_like(g_now)

#                 g_relax = g_now + gamma * (tgt - g_now)

#                 if hasattr(wrapper, 'set_component_gains'):
#                     wrapper.set_component_gains(g_relax)
#                 elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
#                     with torch.no_grad():
#                         wrapper.comp_gain.copy_(g_relax.to(wrapper.comp_gain.device, dtype=wrapper.comp_gain.dtype))

#                 d = (g_relax - g_now).abs()
#                 print(f"[PCA-ROLLBACK] lid={lid} mode={getattr(args,'rollback_to','one')} "
#                       f"gamma={gamma:.2f} |Δg|(min/mean/max)={_stat3(d)} "
#                       f"g_before(min/mean/max)={_stat3(g_now)} g_after(min/mean/max)={_stat3(g_relax)}")

#             if resmix_nudger is not None:
#                 try: resmix_nudger.nudge(step_size=-0.2)
#                 except Exception: pass

#             qv_state['rb_streak'] = 0
#             qv_state['val_acc0']  = val_acc

#     qv_state['val_acc_last'] = val_acc


# # ---------------------------
# # Main
# # ---------------------------
# def main():
#     p=argparse.ArgumentParser()
#     p.add_argument('--task',type=str,default='textcls', choices=['vision','textcls','tokcls'])
#     p.add_argument('--dataset', type=str, default='',
#                   choices=['sst2','mnist','c10','c100'],
#                   help='Universal dataset; if "sst2" routes to text pipeline.')
#     p.add_argument('--text_dataset',type=str,default='sst2', choices=['sst2','ag_news','trec6','conll2003'])
#     p.add_argument('--data_root', type=str, default='data', help='Root for torchvision datasets')
#     p.add_argument('--max_len',type=int,default=128)  # for tokenizer
#     p.add_argument('--batch',type=int,default=64)
#     p.add_argument('--epochs',type=int,default=10)
#     p.add_argument('--seed',type=int,default=42)
#     p.add_argument('--model',type=str,default='distillbert_hf_gate',
#                   choices=list(MODEL_MAP.keys()))
#     p.add_argument('--num_labels',type=int,default=2)
#     p.add_argument('--insert_layers',type=str,default='last')
#     p.add_argument('--weight_lr',type=float,default=3e-4)
#     p.add_argument('--adam_eps',type=float,default=1e-8)
#     p.add_argument('--adam_wd',type=float,default=0.0)

#     # modules
#     p.add_argument('--gate_enable',action='store_true')
#     p.add_argument('--resmix_enable',action='store_true')
#     p.add_argument('--resmix_from_dca_lr',type=float,default=0.0)

#     # filters / fft
#     p.add_argument('--filter_backend',type=str,default='lp_fixed', choices=['darts','lp_fixed','none'])
#     p.add_argument('--lp_k',type=int,default=5)
#     p.add_argument('--gate_tau',type=float,default=1.0)
#     p.add_argument('--feature_mask', action='store_true')
#     p.add_argument('--ff_gamma',type=float,default=1.0)
#     p.add_argument('--ff_amin',type=float,default=0.8)
#     p.add_argument('--ff_apply_on',type=str,default='input', choices=['input','output'])
#     p.add_argument('--ff_use_drv_gate',action='store_true')

#     # DCA
#     p.add_argument('--dca_enable',action='store_true')
#     p.add_argument('--dca_mode', type=str, default='e_ctr_resmix', choices=['e_ctr_resmix'])
#     p.add_argument('--dca_beta',type=float,default=1.0)
#     p.add_argument('--dca_w_lr',type=float,default=0.0)

#     # derivative gate
#     p.add_argument('--dca_derivative_enable',action='store_true')
#     p.add_argument('--dca_drv_mode',type=str,default='ema', choices=['ema','window'])
#     p.add_argument('--dca_drv_lambda',type=float,default=0.2)
#     p.add_argument('--dca_drv_kappa',type=float,default=5.0)

#     # energy-guided
#     p.add_argument('--energy_guided', action='store_true')
#     p.add_argument('--energy_gamma', type=float, default=1.0)
#     p.add_argument('--energy_eps', type=float, default=1e-8)

#     # PCA denoise
#     p.add_argument('--pca_enable',action='store_true')
#     p.add_argument('--pca_k',type=int,default=32)
#     p.add_argument('--pca_lambda',type=float,default=0.8)
#     p.add_argument('--pca_amin',type=float,default=0.85)
#     p.add_argument('--pca_patience',type=int,default=2)

#     # quickval selector
#     p.add_argument('--arm_pick', type=str, default='prob', choices=['prob','softmix','deterministic'])
#     p.add_argument('--arm_temp', type=float, default=1.0)
#     p.add_argument('--arm_weights', type=str, default='pca=1,fft=1,lp=1')
#     p.add_argument('--arm_thr', type=float, default=0.2)
#     p.add_argument('--arm_invpow', type=float, default=2.0)
#     p.add_argument('--arm_eps', type=float, default=1e-6)

#     # hard exploration inside PCA (optional)
#     p.add_argument('--arm_hard', action='store_true')
#     p.add_argument('--arm_hard_mode', type=str, default='bernoulli', choices=['bernoulli', 'topk'])
#     p.add_argument('--arm_hard_topk', type=int, default=0)
#     p.add_argument('--arm_minprob', type=float, default=0.05)
#     p.add_argument('--arm_maxprob', type=float, default=0.95)
#     p.add_argument('--arm_prob_scale', type=float, default=1.0)
#     p.add_argument('--arm_scope', type=str, default='both', choices=['comp', 'res', 'both'])

#     # logging & quickval
#     p.add_argument('--log_every',type=int,default=50)
#     p.add_argument('--quickval_every',type=int,default=400)
#     p.add_argument('--arch_val_batches',type=int,default=2)

#     # spec reg (optional)
#     p.add_argument('--spec_reg_enable',action='store_true')
#     p.add_argument('--spec_reg_lambda',type=float,default=1e-4)
#     p.add_argument('--spec_reg_power',type=float,default=1.0)

#     # io
#     p.add_argument('--log_dir',type=str,default='logs')
#     p.add_argument('--run_tag',type=str,default='run')
#     p.add_argument('--save_dir',type=str,default='checkpoints')
#     p.add_argument('--save_name',type=str,default='model.pt')

#     # SwanLab & CSV
#     p.add_argument('--swan_enable', action='store_true')
#     p.add_argument('--swan_project', type=str, default='default')
#     p.add_argument('--swan_experiment', type=str, default='')
#     p.add_argument('--swan_dir', type=str, default='swan_logs')
#     p.add_argument('--csv_log', action='store_true')
#     p.add_argument('--csv_path', type=str, default='optimization/log/metrics.csv')

#     # Strict base
#     p.add_argument('--strict_base', action='store_true',
#                    help='Disable ALL optional inserts (gates/filters/PCA/ResMix) for a clean base.')

#     p.add_argument('--trigger_train_acc', type=float, default=0.90,
#               help='Only consider denoising when train acc ≥ this.')
#     p.add_argument('--drift_patience', type=int, default=2,
#                 help='Consecutive quickval steps to confirm drift/stagnation.')
#     p.add_argument('--rollback_patience', type=int, default=2,
#                 help='If post-trigger val keeps worsening for N steps, rollback.')
#     p.add_argument('--stagnation_tol', type=float, default=1e-4,
#                 help='Tolerance for valid “no improvement”.')

#     # NEW: 回退参数
#     p.add_argument('--rollback_gamma', type=float, default=0.6,
#                 help='Rollback strength γ in g <- g + γ (tgt - g)')
#     p.add_argument('--rollback_to', type=str, default='one',
#                 choices=['one', 'snapshot'],
#                 help='Rollback target: 1.0 or pre-trigger snapshot')


#     # ---- trigger modes ----
#     p.add_argument('--trigger_mode', type=str, default='worsen',
#                 choices=['worsen', 'stagnate', 'gap'],
#                 help='worsen=恶化触发(默认, OverfitGuard); stagnate=停滞触发; gap=以train-val精度差触发')
#     p.add_argument('--stagnation_patience', type=int, default=None,
#                 help='stagnate模式的停滞耐心；不设则回退到 --drift_patience')
#     p.add_argument('--gap_thr', type=float, default=0.10,
#                 help='gap模式的阈值: 当 train_acc_ema - val_acc ≥ gap_thr 时触发')


#     # Vision 数据集选择（已在统一 dataset 里处理）

#     # 强制 quickval 步应用 PCA gains（绕过 OverfitGuard）
#     p.add_argument('--pca_force', action='store_true',
#                 help='Always apply PCA gains at quickval steps (bypass OverfitGuard).')


#     p.add_argument('--val_split', type=float, default=0.1,
#                help='Fraction of TRAIN used for validation when a train split exists.')
#     p.add_argument('--global_train_frac', type=float, default=0.8,
#                 help='If only a single full dataset exists: fraction used as train-pool (rest is test).')
#     p.add_argument('--global_train_inner_frac', type=float, default=0.9,
#                 help='Inside the train-pool: fraction used for train vs valid.')
#     # 你已有 --seed，这里会用于可复现切分


#     args=p.parse_args()



#       # ---------- NEW: EMA tools & quickval/rollback state ----------
#     class _EMA:
#         def __init__(self, decay=0.9):
#             self.decay = decay
#             self.v = None
#         def update(self, x: float) -> float:
#             x = float(x)
#             self.v = x if self.v is None else (self.decay * self.v + (1 - self.decay) * x)
#             return self.v

#     qv_state = {
#         'train_acc_ema': _EMA(decay=0.9),
#         'train_ce_ema':  _EMA(decay=0.9),
#         'val_acc_last':  None,   # 最近一次 quickval 的 val_acc
#         'armed_live':    False,  # 本轮是否已触发
#         'val_acc0':      None,   # 触发时记录的基线
#         'rb_streak':     0,      # 回退观察计数

#         'best_val_acc':  None,   # stagnate模式用：历史最好
#         'stg_streak':    0,      # stagnate模式用：停滞计数

#     }

#     # ---- dataset/task resolver & num_labels override ----
#     ds = (args.dataset or '').strip().lower()
#     if ds == 'sst2':
#         if args.task != 'textcls':
#             print(f"[INFO] Auto switching task to textcls for dataset={ds}")
#         args.task = 'textcls'
#         args.text_dataset = 'sst2'
#         if args.num_labels != 2:
#             print(f"[INFO] override num_labels: {args.num_labels} -> 2 for sst2")
#         args.num_labels = 2
#     elif ds in ('mnist','c10','cifar10','c100','cifar100'):
#         if args.task != 'vision':
#             print(f"[INFO] Auto switching task to vision for dataset={ds}")
#         args.task = 'vision'
#         ncls = 10 if ds in ('mnist','c10','cifar10') else 100
#         if args.num_labels != ncls:
#             print(f"[INFO] override num_labels: {args.num_labels} -> {ncls} for {ds}")
#         args.num_labels = ncls
#         if ds == 'cifar10': args.dataset = 'c10'
#         if ds == 'cifar100': args.dataset = 'c100'
#     else:
#         # no unified dataset specified: keep user's task/text_dataset
#         pass

#     # --- SwanLab init (optional) ---
#     global swan, csvlog
#     if args.swan_enable:
#         try:
#             import swanlab
#             swan = swanlab.init(
#                 project=args.swan_project,
#                 name=(args.swan_experiment or args.run_tag),
#                 work_dir=args.swan_dir,
#                 config=vars(args)
#             )
#             print("[SWAN] logging enabled.")
#         except Exception as e:
#             print(f"[SWAN] init failed: {e}")
#             swan = None

#     # --- CSV logger (optional) ---
#     if args.csv_log:
#         import csv
#         os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
#         class _CSVLogger:
#             def __init__(self, path):
#                 self.path = path
#                 self._header_written = os.path.exists(path) and os.path.getsize(path) > 0
#             def log(self, row: dict):
#                 hdr = list(row.keys())
#                 need_header = not self._header_written
#                 with open(self.path, 'a', newline='') as f:
#                     w = csv.DictWriter(f, fieldnames=hdr)
#                     if need_header:
#                         w.writeheader()
#                         self._header_written = True
#                     w.writerow(row)
#         csvlog = _CSVLogger(args.csv_path)
#         print(f"[CSV] logging to {args.csv_path}")

#     set_seed(args.seed)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # --- build model ---
#     ModelCls = MODEL_MAP[args.model]

#     # 组装构造参数
#     extra_kwargs = {}

#     # 文本任务：传给小模型
#     if args.task == 'textcls' and args.model in ('small_bert', 'small_transformer', 'distillbert_hf_gate'):
#         extra_kwargs['text_task'] = 'cls'
#         extra_kwargs['text_num_classes'] = args.num_labels

#     # 视觉任务：仅当构造器接受 num_classes 时才传，避免不兼容
#     if args.task == 'vision':
#         import inspect
#         sig = inspect.signature(ModelCls.__init__)
#         if 'num_classes' in sig.parameters:
#             extra_kwargs['num_classes'] = args.num_labels

#     model = ModelCls(
#         insert_layers=args.insert_layers,
#         filter_backend=args.filter_backend, lp_k=args.lp_k,
#         gate_tau=args.gate_tau,
#         use_gate=args.gate_enable, use_resmix=args.resmix_enable,
#         pca_enable=args.pca_enable, pca_k=args.pca_k, pca_amin=args.pca_amin,
#         feature_mask=args.feature_mask,
#         ff_gamma=args.ff_gamma, ff_amin=args.ff_amin,
#         ff_apply_on=args.ff_apply_on, ff_use_drv_gate=args.ff_use_drv_gate,
#         **extra_kwargs
#     ).to(device)

#     # === strict_base: 真空化 ===
#     if args.strict_base and hasattr(model, 'disable_all_inserts'):
#         print("[INFO] Strict base ON: disabling all inserts/gates/filters/PCA/ResMix.")
#         model.disable_all_inserts()
#         args.quickval_every = 0
#         args.arch_val_batches = 0

#     # 兜底：如果构造参数没有生效，这里强制把分类头改成 num_labels（仅视觉）
#     if args.task == 'vision':
#         import torch.nn as nn
#         replaced = False
#         if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
#             in_f = model.classifier.in_features
#             model.classifier = nn.Linear(in_f, args.num_labels).to(device)
#             replaced = True
#         elif hasattr(model, 'head') and hasattr(model.head, 'in_features'):
#             in_f = model.head.in_features
#             model.head = nn.Linear(in_f, args.num_labels).to(device)
#             replaced = True
#         elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
#             in_f = model.fc.in_features
#             model.fc = nn.Linear(in_f, args.num_labels).to(device)
#             replaced = True

#         if replaced:
#             print(f"[FIX] force reset classifier head -> num_classes={args.num_labels}")

#     # --- optim ---
#     opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
#                             lr=args.weight_lr, eps=args.adam_eps, weight_decay=args.adam_wd)

#     # --- DCA init & hooks ---
#     dca = None
#     if args.dca_enable and (DCA is not None):
#         if DCACfg is not None:
#             dca_cfg = DCACfg(enable=True)
#             dca = DCA(cfg=dca_cfg, num_classes=args.num_labels)
#         else:
#             try:
#                 dca = DCA(num_classes=args.num_labels)
#             except TypeError:
#                 try:
#                     dca = DCA()
#                 except Exception as e:
#                     print(f"[WARN] DCA construction failed: {e}")
#                     dca = None
#         if dca is not None and hasattr(dca, "attach_hooks"):
#             if not getattr(dca, "_attached", False):
#                 dca.attach_hooks(model); dca._attached = True; print("[DCA] hooks attached")

#     dca_drv = None
#     if args.dca_derivative_enable and (DCADerivative is not None):
#         try:
#             dca_drv = DCADerivative()
#         except Exception as e:
#             print(f"[WARN] DCADerivative() failed, disable derivative gate: {e}")
#             dca_drv = None

#     resmix_nudger = SoftResMixNudger(model) if args.resmix_enable and args.resmix_from_dca_lr > 0 else None

#     # PCA manager (disabled in strict_base)
#     pca_mgr = None
#     if args.pca_enable and (not args.strict_base):
#         pca_mgr = PCAOverlapManager(k=args.pca_k, ema_decay=0.99, device=str(device))
#         if hasattr(model, 'attach_pca_manager'):
#             model.attach_pca_manager(pca_mgr)

#     guard = OverfitGuard(OverfitGuardConfig(patience=args.pca_patience,
#                                             kappa=args.dca_drv_kappa, clip=1.0))

#     # ---- loaders ----
#     # collate = getattr(model, 'collate_fn', None)

#     # if args.task == 'textcls':
#     #     # 仅实现了 SST-2
#     #     if (args.dataset == 'sst2') or (args.text_dataset == 'sst2'):
#     #         train_loader, val_loader = build_sst2_loaders(args.batch, collate)
#     #     else:
#     #         raise ValueError(
#     #             f"Only 'sst2' is implemented for textcls. Got --dataset='{args.dataset}', --text_dataset='{args.text_dataset}'"
#     #         )

#     # elif args.task == 'vision':
#     #     if args.dataset not in ('mnist', 'c10', 'c100'):
#     #         raise ValueError("When --task vision, please set --dataset in ['mnist','c10','c100'].")

#     #     # 安全保护：文本模型不要跑视觉数据
#     #     clsname = model.__class__.__name__.lower()
#     #     if hasattr(model, 'bert') or 'bert' in clsname or 'distill' in clsname:
#     #         print(f"[ERROR] Model '{model.__class__.__name__}' is a TEXT model; cannot run on VISION dataset '{args.dataset}'.")
#     #         sys.exit(2)

#     #     # train_loader, val_loader = build_vision_loaders(args.dataset, args.batch, args.data_root)
#     #     train_loader, val_loader, _ = build_vision_loaders(args.dataset, args.batch)

#     # else:
#     #     raise ValueError(f"Unsupported task: {args.task}")


#     # # ---- loaders ----
#     # collate = getattr(model, 'collate_fn', None)

#     # if args.task == 'textcls':
#     #     # 这里只实现了 SST-2
#     #     if (args.dataset == 'sst2') or (args.text_dataset == 'sst2'):
#     #         # 注意：你在步骤 1 已把 build_sst2_loaders 改成返回 (train, val, test=None)
#     #         train_loader, val_loader, test_loader = build_sst2_loaders(args.batch, collate)
#     #     else:
#     #         raise ValueError(
#     #             f"Only 'sst2' is implemented for textcls. Got --dataset='{args.dataset}', --text_dataset='{args.text_dataset}'"
#     #         )

#     # elif args.task == 'vision':
#     #     if args.dataset not in ('mnist', 'c10', 'c100'):
#     #         raise ValueError("When --task vision, please set --dataset in ['mnist','c10','c100'].")
#     #     clsname = model.__class__.__name__.lower()
#     #     if hasattr(model, 'bert') or 'bert' in clsname or 'distill' in clsname:
#     #         print(f"[ERROR] Model '{model.__class__.__name__}' is a TEXT model; cannot run on VISION dataset '{args.dataset}'.")
#     #         sys.exit(2)
#     #     # 注意：你在步骤 1 已把 build_vision_loaders 改成返回 (train, val, test, num_labels)
#     #     train_loader, val_loader, test_loader, _ = build_vision_loaders(args.dataset, args.batch)

#     # else:
#     #     raise ValueError(f"Unsupported task: {args.task}")



#     # ---- loaders ----
#     collate = getattr(model, 'collate_fn', None)

#     if args.task == 'textcls':
#         # 目前只实现 SST-2：官方 validation -> test；valid 从 train 切
#         if (args.dataset == 'sst2') or (args.text_dataset == 'sst2'):
#             train_loader, val_loader, test_loader = build_sst2_loaders(
#                 args.batch,
#                 collate_fn=collate,
#                 val_split=args.val_split,   # <- 从 train 里切 valid 的比例
#                 seed=args.seed              # <- 固定随机种子，切分可复现
#             )
#         else:
#             raise ValueError(
#                 f"Only 'sst2' is implemented for textcls. Got --dataset='{args.dataset}', --text_dataset='{args.text_dataset}'"
#             )

#     elif args.task == 'vision':
#         # 有官方 test 的视觉数据集：test 用官方，valid 从 train 切
#         if args.dataset not in ('mnist', 'c10', 'c100'):
#             raise ValueError("When --task vision, please set --dataset in ['mnist','c10','c100'].")
#         clsname = model.__class__.__name__.lower()
#         if hasattr(model, 'bert') or 'bert' in clsname or 'distill' in clsname:
#             print(f"[ERROR] Model '{model.__class__.__name__}' is a TEXT model; cannot run on VISION dataset '{args.dataset}'.")
#             sys.exit(2)

#         train_loader, val_loader, test_loader, _ = build_vision_loaders(
#             name=args.dataset,
#             batch=args.batch,
#             root=args.data_root,           # <- 数据根目录
#             val_split=args.val_split,      # <- 从 train 里切 valid 的比例
#             seed=args.seed                 # <- 固定随机种子
#         )
#     else:
#         raise ValueError(f"Unsupported task: {args.task}")


#     # # --- train loop ---
#     # state = {'step': 0, 'best_acc': -1}

#     state = {'step': 0, 'best_acc': -1, 'last_qv_step': 0}
#     for epoch in range(1, args.epochs + 1):
#         m = run_epoch(model, train_loader, opt, device, state, args, train=True,  pca_mgr=pca_mgr)
#         print(f"[E{epoch:02d}] train loss={m['loss']:.4f} acc={m['acc']:.4f}")

#         model._last_train_acc_epoch = float(m['acc'])  # NEW: 提供给 quickval 触发门槛

#                 # ---------- NEW: update train-side EMA proxies ----------
#         qv_state['train_acc_ema'].update(float(m['acc']))
#         # 如果没有显式 CE，可用 1-acc 作为粗略 proxy：
#         qv_state['train_ce_ema'].update(float(m.get('loss', 1.0 - m['acc'])))



#         if swan is not None:
#             swan.log({"train/epoch_loss": float(m['loss']), "train/epoch_acc": float(m['acc'])}, step=epoch)
#         if csvlog is not None:
#             csvlog.log({"stage": "train_epoch", "epoch": epoch,
#                         "loss": round(float(m['loss']), 6), "acc": round(float(m['acc']), 6)})

#         # ---- 如果存在 test_loader，则在每个 epoch 进行测试并写日志 ----
#         if test_loader is not None:
#             tm = evaluate(model, test_loader, device, args)
#             print(f"[E{epoch:02d}]  test loss={tm['loss']:.4f} acc={tm['acc']:.4f}")
#             if swan is not None:
#                 swan.log({"test/epoch_loss": float(tm['loss']), "test/epoch_acc": float(tm['acc'])}, step=epoch)
#             if csvlog is not None:
#                 csvlog.log({"stage":"test_epoch","epoch":epoch,
#                             "loss":round(float(tm['loss']),6),"acc":round(float(tm['acc']),6)})


#         # if (args.quickval_every > 0) and (not args.strict_base) and (state['step'] % args.quickval_every == 0):
#         #     opt.zero_grad(set_to_none=True)
#         #     # quickval_arch_step(model, val_loader, device, args, dca, dca_drv, resmix_nudger, pca_mgr, guard)
#         #     quickval_arch_step(model, val_loader, device, args, dca, dca_drv, resmix_nudger, pca_mgr, guard, qv_state)

#         # vm = evaluate(model, val_loader, device, args)
#         # print(f"[E{epoch:02d}] valid loss={vm['loss']:.4f} acc={vm['acc']:.4f}")
#         # if swan is not None:
#         #     swan.log({"valid/epoch_loss": float(vm['loss']), "valid/epoch_acc": float(vm['acc'])}, step=epoch)
#         # if csvlog is not None:
#         #     csvlog.log({"stage": "valid_epoch", "epoch": epoch,
#         #                 "loss": round(float(vm['loss']), 6), "acc": round(float(vm['acc']), 6)})

#         # 差值触发 quickval（不再依赖整除）
#         if (args.quickval_every > 0) and (not args.strict_base):
#             if state['step'] - state.get('last_qv_step', 0) >= args.quickval_every:
#                 opt.zero_grad(set_to_none=True)
#                 print(f"[QVAL] step={state['step']} (Δ={state['step']-state.get('last_qv_step',0)}) epoch={epoch} -> quickval")
#                 quickval_arch_step(model, val_loader, device, args,
#                                 dca, dca_drv, resmix_nudger, pca_mgr, guard, qv_state)
#                 state['last_qv_step'] = state['step']

#         vm = evaluate(model, val_loader, device, args)
#         print(f"[E{epoch:02d}] valid loss={vm['loss']:.4f} acc={vm['acc']:.4f}")
#         if swan is not None:
#             swan.log({"valid/epoch_loss": float(vm['loss']), "valid/epoch_acc": float(vm['acc'])}, step=epoch)
#         if csvlog is not None:
#             csvlog.log({"stage": "valid_epoch", "epoch": epoch,
#                         "loss": round(float(vm['loss']), 6), "acc": round(float(vm['acc']), 6)})



#         if vm['acc'] > state['best_acc']:
#             state['best_acc'] = vm['acc']
#             os.makedirs(args.save_dir, exist_ok=True)
#             save_path = os.path.join(args.save_dir, args.save_name)
#             torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch}, save_path)
#             print(f"[E{epoch:02d}] saved best to {save_path}")

# if __name__=='__main__':
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified train_eval_final.py  (Reproducible Shuffle Every Epoch + Deterministic CUDA)
- Text (SST-2) and Vision (MNIST / CIFAR-10 / CIFAR-100)
- PCA/FFT denoise + OverfitGuard + energy-guided nudge
- SwanLab logging + CSV logging
- Strict base switch to disable all inserts
- [B] Reproducibility: one global Generator reused by all train DataLoaders
"""


import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated.*",
    category=FutureWarning
)

import argparse, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset  # for text datasets (SST-2)
# 必须在 import torch 之前设置，确保 cublas 使用确定性工作区


# ---------------------------
# Globals for logging backends
# ---------------------------
swan = None     # SwanLab run handle
csvlog = None   # CSV logger handle

# --- optional: transformers tokenizer for raw text ---
_TOKENIZER = None
def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        except Exception as e:
            raise RuntimeError(
                f"Need transformers to tokenize raw text for small_bert. Install it or "
                f"provide a model.collate_fn. Error: {e}"
            )
    return _TOKENIZER

# --- DCA & friends (robust imports) ---
try:
    from modules.dca import DCACfg, DCA
except Exception as e:
    DCACfg = None; DCA = None
    print(f"[WARN] DCA unavailable: {e}")

try:
    from modules.dca_derivative import DCADerivative
except Exception as e:
    DCADerivative = None
    print(f"[WARN] DCADerivative unavailable: {e}")

try:
    from modules.dca_e_ctr_resmix import build_e_ctr_resmix
except Exception as e:
    build_e_ctr_resmix = None
    print(f"[WARN] build_e_ctr_resmix unavailable: {e}")

try:
    from modules.spec_reg import spectral_penalty_depthwise
except Exception:
    spectral_penalty_depthwise = None

from modules.soft_resmix import SoftResMixNudger
from modules.pca_overlap import PCAOverlapManager
from modules.overfit_guard import OverfitGuard, OverfitGuardConfig

# --- models ---
from models.distillbert_hf_gate import DistillBertHFGate
from models.small_bert import SmallBERT
from models.small_transformer import SmallTransformer

# vision dataloaders
try:
    from torchvision import datasets as tvds, transforms as T
except Exception:
    tvds = None; T = None

MODEL_MAP = {
    'distillbert_hf_gate': DistillBertHFGate,
    'small_bert': SmallBERT,
    'small_transformer': SmallTransformer,
}

# =========================
# [B] Reproducibility utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强可复现（可能稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# def enforce_cuda_determinism():
#     """关闭 TF32 和非确定性的 SDPA/Flash 注意力实现。"""
#     try:
#         torch.backends.cuda.matmul.allow_tf32 = False
#         torch.backends.cudnn.allow_tf32 = False
#     except Exception:
#         pass
#     try:
#         # PyTorch 2.1+
#         if hasattr(torch.backends.cuda, "enable_flash_sdp"):
#             torch.backends.cuda.enable_flash_sdp(False)
#             torch.backends.cuda.enable_mem_efficient_sdp(False)
#             torch.backends.cuda.enable_math_sdp(True)   # 使用确定性的 math kernel
#         # PyTorch 2.0
#         if hasattr(torch.backends.cuda, "sdp_kernel"):
#             torch.backends.cuda.sdp_kernel(enable_flash=False,
#                                            enable_mem_efficient=False,
#                                            enable_math=True)
#     except Exception:
#         pass
#     # CuBLAS 需要环境变量，请在 shell 里设置：
#     if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
#         print("[DETERMINISM] Tip: export CUBLAS_WORKSPACE_CONFIG=:4096:8 "
#               "(set this in your shell before running to silence CuBLAS warnings)")

def enforce_cuda_determinism():
    # 关闭 TF32（避免 matmul 走 TF32）
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        # PyTorch 2.0+：阻止 Flash/Memory-Efficient SDPA，强制 math kernel（确定性）
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass
    try:
        # 进一步把 matmul 精度锁到 FP32（避免自动选 TF32）
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def make_global_train_generator(seed: int) -> torch.Generator:
    """
    创建一次并复用的全局训练 generator。
    复跑时（同 seed、同代码路径）能保证每个 epoch 的 shuffle 序列一致；
    同一次运行里，随着 DataLoader 每轮迭代，该 generator 的内部状态会前进，
    因而每个 epoch 的顺序不同（满足“每轮都打乱”）。
    """
    return torch.Generator().manual_seed(int(seed))

def seed_worker_fn(_):
    """
    DataLoader worker 的初始化函数。
    将 numpy/random 的种子与 PyTorch 的 initial_seed 对齐，保证 transforms 的随机也可复现。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------------------
# Data: text (SST-2)
# ---------------------------
def build_sst2_loaders(batch, collate_fn=None, val_split=0.1, seed=42,
                       num_workers=0, pin_memory=False,
                       train_generator=None,              # 训练的全局 generator
                       val_shuffle=True, val_generator=None):   # 验证的独立 generator
    """
    [B] 训练集：shuffle=True + 复用 train_generator
    验证/测试：val_shuffle 控制 + val_generator；测试固定不 shuffle
    """
    ds = load_dataset("glue", "sst2")
    train_full = list(zip(ds['train']['sentence'], ds['train']['label']))
    test_full  = list(zip(ds['validation']['sentence'], ds['validation']['label']))

    n_total = len(train_full)
    n_val   = max(1, int(round(n_total * float(val_split))))
    g_split = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g_split).tolist()
    val_idx   = set(perm[:n_val])
    train_set = [train_full[i] for i in range(n_total) if i not in val_idx]
    val_set   = [train_full[i] for i in perm[:n_val]]

    train_loader = DataLoader(
        train_set, batch_size=batch, shuffle=True, drop_last=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=seed_worker_fn,
        generator=train_generator
    )
    val_loader   = DataLoader(
        val_set, batch_size=batch,
        shuffle=bool(val_shuffle),
        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory,
        generator=val_generator
    )
    test_loader  = DataLoader(
        test_full, batch_size=batch, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader

# --- vision ---
def build_vision_loaders(name: str, batch: int, root: str="data",
                         val_split: float=0.1, seed: int=42,
                         num_workers=4, pin_memory=True,
                         train_generator=None,              # 训练的全局 generator
                         val_shuffle=True, val_generator=None):   # 验证的独立 generator
    """
    [B] 训练集：shuffle=True + 复用 train_generator
    验证/测试：val_shuffle 控制 + val_generator；测试固定不 shuffle
    """
    assert tvds is not None and T is not None, "Need torchvision for vision datasets"

    if name == 'mnist':
        tr = T.Compose([T.ToTensor()])
        te = T.Compose([T.ToTensor()])
        train_full = tvds.MNIST(root, train=True,  download=True, transform=tr)
        test       = tvds.MNIST(root, train=False, download=True, transform=te)
        num_labels = 10

    elif name in ('c10', 'cifar10'):
        tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        te = T.Compose([T.ToTensor()])
        train_full = tvds.CIFAR10(root, train=True,  download=True, transform=tr)
        test       = tvds.CIFAR10(root, train=False, download=True, transform=te)
        num_labels = 10

    elif name in ('c100', 'cifar100'):
        tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        te = T.Compose([T.ToTensor()])
        train_full = tvds.CIFAR100(root, train=True,  download=True, transform=tr)
        test       = tvds.CIFAR100(root, train=False, download=True, transform=te)
        num_labels = 100

    else:
        raise ValueError(f"Unknown vision dataset: {name}")

    # 从训练集里切 valid（切分使用独立的 generator，确保可复现）
    n_total = len(train_full)
    n_val   = max(1, int(round(n_total * float(val_split))))
    g_split = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g_split).tolist()
    val_idx = set(perm[:n_val])

    from torch.utils.data import Subset
    train_ds = Subset(train_full, [i for i in range(n_total) if i not in val_idx])
    val_ds   = Subset(train_full, [i for i in perm[:n_val]])

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=seed_worker_fn,
        generator=train_generator
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch,
        shuffle=bool(val_shuffle),
        num_workers=0, pin_memory=pin_memory,
        generator=val_generator
    )
    test_loader  = DataLoader(
        test, batch_size=batch, shuffle=False,
        num_workers=0, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader, num_labels

# ---------------------------
# Utils
# ---------------------------
def accuracy(logits, labels):
    preds = logits.argmax(-1)
    return (preds == labels).float().mean().item()

# Robust batch parser + auto-tokenize
def _maybe_tokenize_text(x_like, max_len, device):
    if isinstance(x_like, (list, tuple)) and len(x_like) > 0 and isinstance(x_like[0], str):
        tok = _get_tokenizer()
        enc = tok(list(x_like), padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)
    if isinstance(x_like, str):
        tok = _get_tokenizer()
        enc = tok([x_like], padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)
    return None, None

def parse_batch(batch, device, max_len=128):
    def to_dev(t):
        return t.to(device) if torch.is_tensor(t) else t

    def norm_x(xlike):
        if isinstance(xlike, dict):
            xi = xlike.get('input_ids', xlike.get('x', xlike.get('inputs')))
            xm = xlike.get('attention_mask', xlike.get('mask'))
            ids, am = _maybe_tokenize_text(xi, max_len, device)
            if ids is not None: return ids, am
            if (xm is None) and isinstance(xi, (tuple, list)) and len(xi) == 2:
                xi, xm = xi
            return to_dev(xi), to_dev(xm)
        if isinstance(xlike, (tuple, list)):
            if len(xlike) == 2 and torch.is_tensor(xlike[0]):
                return to_dev(xlike[0]), to_dev(xlike[1])
            ids, am = _maybe_tokenize_text(xlike, max_len, device)
            if ids is not None: return ids, am
            if len(xlike) >= 1: return to_dev(xlike[0]), None
        if torch.is_tensor(xlike):
            return to_dev(xlike), None
        ids, am = _maybe_tokenize_text(xlike, max_len, device)
        if ids is not None: return ids, am
        return xlike, None

    if isinstance(batch, dict):
        xi, xm = norm_x(batch)
        y = batch.get('labels', batch.get('y', None))
        return xi, xm, to_dev(y)

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            a, b = batch; xi, xm = norm_x(a); y = b
            return xi, xm, to_dev(y)
        if len(batch) == 3:
            xi, xm = norm_x(batch[0])
            xm = batch[1] if (xm is None) else xm
            y = batch[2]
            return xi, to_dev(xm), to_dev(y)
        xi, xm = norm_x(batch[0])
        if xm is None and len(batch) > 2: xm = batch[1]
        y = batch[-1]
        return xi, to_dev(xm), to_dev(y)

    if torch.is_tensor(batch):
        return batch.to(device), None, None

    ids, am = _maybe_tokenize_text(batch, max_len, device)
    if ids is not None: return ids, am, None

    raise TypeError(f"Unsupported batch type: {type(batch)}")

def call_model(model, x_ids, attn_mask):
    if hasattr(model, 'bert') or model.__class__.__name__.lower().startswith('distill'):
        return model(x_ids, attention_mask=attn_mask)
    try:
        return model(x_ids, attn_mask=attn_mask) if attn_mask is not None else model(x_ids)
    except TypeError:
        return model(x_ids)

# ---------------------------
# Train / Eval
# ---------------------------
def run_epoch(model, loader, opt, device, state, args, train=True, pca_mgr=None):
    model.train(train)
    tot_loss=tot_acc=n=0
    for batch in loader:
        if hasattr(model, 'collate_fn'):
            batch = model.collate_fn(batch)
        x_ids, attn_mask, labels = parse_batch(batch, device, max_len=args.max_len)
        assert torch.is_tensor(x_ids), f"x_ids must be Tensor, got {type(x_ids)}"

        if train: opt.zero_grad(set_to_none=True)
        out = call_model(model, x_ids, attn_mask)
        logits = out['logits']
        assert logits.dim() == 2 and logits.size(1) == args.num_labels, \
            f"expect logits [B,{args.num_labels}], got {tuple(logits.shape)}"

        if labels is None:
            continue
        ce_now = out.get('loss') or F.cross_entropy(logits, labels)
        if train:
            with torch.no_grad():
                model._last_train_ce = ce_now.detach()
        loss = ce_now

        # 持续更新 PCA 的 train 统计
        if train and getattr(args, 'pca_enable', False) and hasattr(model, 'pca_wrappers') and (pca_mgr is not None):
            for lid, wrap in getattr(model, 'pca_wrappers', {}).items():
                Xtr = getattr(wrap, '_last_input', None)
                if Xtr is None: continue
                if Xtr.dim() == 3: Xt = Xtr.reshape(-1, Xtr.shape[-1])
                elif Xtr.dim() == 2: Xt = Xtr
                else: continue
                try: pca_mgr.update_train(lid, Xt)
                except Exception as e:
                    print(f"[PCA][update_train] lid={lid} warn: {e}")

        acc = accuracy(logits, labels)
        if train:
            loss.backward(); opt.step(); state['step']+=1
        bs=labels.size(0); tot_loss+=loss.item()*bs; tot_acc+=acc*bs; n+=bs

        if train and args.log_every>0 and state['step']%args.log_every==0:
            mean_loss=tot_loss/max(1,n); mean_acc=tot_acc/max(1,n)
            print(f"[S{state['step']}] train loss={mean_loss:.4f} acc={mean_acc:.4f}")
            try:
                if swan is not None:
                    swan.log({"train/step_loss": float(mean_loss),
                              "train/step_acc": float(mean_acc)}, step=state['step'])
                if csvlog is not None:
                    csvlog.log({"stage":"train_step","step":state['step'],"epoch":-1,
                                "loss":round(float(mean_loss),6),"acc":round(float(mean_acc),6)})
            except Exception:
                pass

    return {'loss':tot_loss/max(1,n),'acc':tot_acc/max(1,n)}

@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval(); tot_loss=tot_acc=n=0
    for batch in loader:
        if hasattr(model, 'collate_fn'):
            batch = model.collate_fn(batch)
        x_ids, attn_mask, labels = parse_batch(batch, device, max_len=args.max_len)
        assert torch.is_tensor(x_ids), f"x_ids must be Tensor, got {type(x_ids)}"
        out = call_model(model, x_ids, attn_mask)
        logits = out['logits']
        assert logits.dim() == 2 and logits.size(1) == args.num_labels, \
            f"expect logits [B,{args.num_labels}], got {tuple(logits.shape)}"

        loss = out.get('loss') or F.cross_entropy(logits, labels)
        acc = accuracy(logits, labels)
        bs=labels.size(0); tot_loss+=loss.item()*bs; tot_acc+=acc*bs; n+=bs
        # （注意：val/test loader 的 shuffle 行为在构建时已经按开关处理）
    return {'loss':tot_loss/max(1,n),'acc':tot_acc/max(1,n)}

# ---------- quickval_arch_step ----------
def quickval_arch_step(model, val_loader, device, args, dca, dca_drv, resmix_nudger, pca_mgr, guard, qv_state):
    model.eval()
    import itertools
    val_iter = itertools.islice(iter(val_loader), int(max(1, args.arch_val_batches)))

    def _stat3(t: torch.Tensor):
        t = t.float().view(-1)
        return f"{t.min().item():.4f}/{t.mean().item():.4f}/{t.max().item():.4f}"

    ce_list, acc_list = [], []

    # 1) 小批验证代理
    with torch.no_grad():
        for batch in val_iter:
            if isinstance(batch, (list, tuple)):
                xb, yb = batch[0].to(device), batch[1].to(device)
                fwd_kwargs = {}
            elif isinstance(batch, dict):
                yb = batch.get('labels', batch.get('label'))
                if yb is None:
                    raise RuntimeError("quickval batch dict has no 'labels' key.")
                fwd_kwargs = {k: (v.to(device) if torch.is_tensor(v) else v)
                              for k, v in batch.items() if k not in ['labels', 'label']}
                xb = None
                yb = yb.to(device)
            else:
                raise RuntimeError(f"Unsupported batch type in quickval: {type(batch)}")

            out = model(xb) if (xb is not None and len(fwd_kwargs) == 0) else model(**fwd_kwargs)
            logits = out if isinstance(out, torch.Tensor) else out['logits'] if isinstance(out, dict) else out.logits
            ce = F.cross_entropy(logits, yb).item()
            acc = (logits.argmax(dim=-1) == yb).float().mean().item()
            ce_list.append(ce); acc_list.append(acc)

    num = max(1, len(ce_list))
    val_ce  = float(sum(ce_list) / num)
    val_acc = float(sum(acc_list) / num)

    # 2) 训练侧 proxy + 导数权重
    train_ce_proxy = qv_state['train_ce_ema'].v if qv_state['train_ce_ema'].v is not None else val_ce
    drv_weight = 0.0
    if dca_drv is not None:
        try: drv_weight = float(dca_drv.weight())
        except Exception: drv_weight = 0.0

    # 3) OverfitGuard 判定（分离 train/val proxy）+ 训练精度门控
    armed, guard_w, extras = guard.tick(train_proxy=train_ce_proxy, val_proxy=val_ce)

    # 训练精度门控：只有当 train_acc 的 EMA ≥ 阈值时才允许触发（除非 pca_force）
    train_acc_ema = qv_state['train_acc_ema'].v
    train_acc_gate_ok = (train_acc_ema is not None) and (train_acc_ema >= float(args.trigger_train_acc))

    if not getattr(args, 'pca_force', False) and not train_acc_gate_ok:
        armed = False
        guard_w = 0.0

    print(f"[QVAL] guard_armed={armed} guard_w={guard_w:.3f} "
          f"train_acc_ema={None if train_acc_ema is None else round(train_acc_ema,4)} "
          f"gate_ok={train_acc_gate_ok} drv_w={drv_weight:.3f} "
          f"val_ce={val_ce:.4f} val_acc={val_acc:.4f}")

    # ---------- Trigger mode override ----------
    mode = getattr(args, 'trigger_mode', 'worsen')
    armed_by_mode = False
    mode_weight = 0.0  # 和 guard_w 取 max

    if not getattr(args, 'pca_force', False) and train_acc_gate_ok:
        if mode == 'worsen':
            armed_by_mode = armed
            mode_weight = guard_w

        elif mode == 'stagnate':
            if (qv_state['best_val_acc'] is None) or \
               (val_acc >= qv_state['best_val_acc'] + float(args.stagnation_tol)):
                qv_state['best_val_acc'] = val_acc
                qv_state['stg_streak'] = 0
            else:
                qv_state['stg_streak'] = qv_state.get('stg_streak', 0) + 1

            patience = int(args.stagnation_patience or args.drift_patience)
            if qv_state['stg_streak'] >= patience:
                armed_by_mode = True
                mode_weight = 0.5

        elif mode == 'gap':
            tr_acc = float(qv_state['train_acc_ema'].v or 0.0)
            gap = max(0.0, tr_acc - float(val_acc))
            if gap >= float(args.gap_thr):
                armed_by_mode = True
                mode_weight = min(1.0, max(0.0, (gap - args.gap_thr) / max(1e-6, 1.0 - args.gap_thr)))
        else:
            armed_by_mode = armed
            mode_weight = guard_w

    if getattr(args, 'pca_force', False):
        armed = True
    else:
        armed = armed_by_mode
        guard_w = max(float(guard_w), float(mode_weight))

    print(f"[TRIGGER] mode={mode} armed={armed} guard_w={guard_w:.3f} "
          f"train_acc_ema={(qv_state['train_acc_ema'].v or 0.0):.4f} val_acc={val_acc:.4f}")

    # 4) 找 PCA wrappers（兼容 dict / list / tuple）
    pca_wrappers = []
    if hasattr(model, 'pca_wrappers'):
        w = model.pca_wrappers
        if isinstance(w, dict):
            pca_wrappers = [w[k] for k in sorted(w.keys())]
        elif isinstance(w, (list, tuple)):
            pca_wrappers = list(w)
    elif hasattr(model, 'pca_modules') and isinstance(model.pca_modules, (list, tuple)):
        pca_wrappers = list(model.pca_modules)

    if len(pca_wrappers) == 0 or (pca_mgr is None):
        qv_state['val_acc_last'] = val_acc
        return

    # 5) 触发：应用 soft/hard，并保存 snapshot
    if armed:
        qv_state.setdefault('g_snap', {})
        for l, wrapper in enumerate(pca_wrappers):
            # --- 用验证激活算 overlap -> rho ---
            Xval = getattr(wrapper, '_last_input', None)
            if Xval is None:
                print(f"[PCA] lid={l} skip: no _last_input on wrapper")
                continue

            if Xval.dim() == 3:          # [B, T, D]
                Xv = Xval.reshape(-1, Xval.shape[-1])
            elif Xval.dim() == 2:        # [N, C]
                Xv = Xval
            else:
                print(f"[PCA] lid={l} skip: unsupported X shape {tuple(Xval.shape)}")
                continue

            scores = pca_mgr.overlap_scores(l, Xv)  # -> {'O_i': [k], 'O_res': ...}
            rho = scores['O_i'].float().to(device)

            # 读 g_before & 存快照
            if hasattr(wrapper, 'get_component_gains') and (wrapper.get_component_gains() is not None):
                g_before = wrapper.get_component_gains().detach().clone()
            elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
                g_before = wrapper.comp_gain.detach().clone()
            else:
                g_before = None
            if g_before is not None:
                qv_state['g_snap'][l] = g_before.clone()

            strength = float(max(guard_w, drv_weight))
            a_min = float(getattr(args, 'pca_amin', 0.85))

            if getattr(args, 'arm_hard', False):
                scale   = float(getattr(args, 'arm_prob_scale', 1.0))
                minprob = float(getattr(args, 'arm_minprob', 0.05))
                maxprob = float(getattr(args, 'arm_maxprob', 0.95))
                p_comp  = torch.clamp(scale * (1.0 - rho), min=minprob, max=maxprob)

                mode_h = getattr(args, 'arm_hard_mode', 'bernoulli')
                if mode_h == 'bernoulli':
                    m_comp = torch.bernoulli(p_comp).to(rho.dtype)
                else:
                    score = (1.0 - rho)
                    K_cfg = int(getattr(args, 'arm_hard_topk', 0))
                    if K_cfg > 0:
                        K = min(K_cfg, score.numel())
                    else:
                        K = int(torch.clamp(p_comp.sum().round(), min=1.0).item())
                        K = min(K, score.numel())
                    top_idx = torch.topk(score, k=K, largest=True).indices
                    m_comp = torch.zeros_like(score); m_comp[top_idx] = 1.0

                g_hard = (1.0 - m_comp) + m_comp * a_min
                adj_g  = 1.0 - strength * (1.0 - g_hard)

                if hasattr(wrapper, 'set_component_gains'):
                    wrapper.set_component_gains(adj_g)
                elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
                    with torch.no_grad():
                        wrapper.comp_gain.copy_(adj_g.to(wrapper.comp_gain.device, dtype=wrapper.comp_gain.dtype))
                if hasattr(wrapper, 'set_res_gain'): wrapper.set_res_gain(1.0)
                if hasattr(wrapper, 'set_blend'):    wrapper.set_blend(1.0)

                sel = (m_comp > 0.5).nonzero(as_tuple=False).flatten()
                head = ", ".join([str(int(i)) for i in sel[:12].tolist()]) + (" ..." if sel.numel() > 12 else "")
                if g_before is not None:
                    g_after = adj_g if isinstance(adj_g, torch.Tensor) else torch.as_tensor(adj_g, device=g_before.device, dtype=g_before.dtype)
                    d = (g_after - g_before).abs()
                    print(f"[ARM-HARD] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
                          f"p_comp(min/mean/max)={_stat3(p_comp)} selected={int(sel.numel())}/{m_comp.numel()} "
                          f"indices=[{head}] |Δg|(min/mean/max)={_stat3(d)}")
                else:
                    print(f"[ARM-HARD] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
                          f"p_comp(min/mean/max)={_stat3(p_comp)} selected={int(sel.numel())}/{m_comp.numel()} indices=[{head}]")

            else:
                # SOFT：base_g -> adj_g
                base_g = (1.0 - args.pca_lambda * (1.0 - rho)).clamp(min=a_min, max=1.0)
                adj_g  = 1.0 - strength * (1.0 - base_g)

                if hasattr(wrapper, 'set_component_gains'):
                    wrapper.set_component_gains(adj_g)
                elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
                    with torch.no_grad():
                        wrapper.comp_gain.copy_(adj_g.to(wrapper.comp_gain.device, dtype=wrapper.comp_gain.dtype))
                if hasattr(wrapper, 'set_res_gain'): wrapper.set_res_gain(1.0)
                if hasattr(wrapper, 'set_blend'):    wrapper.set_blend(1.0)

                if g_before is not None:
                    g_after = adj_g if isinstance(adj_g, torch.Tensor) else torch.as_tensor(adj_g, device=g_before.device, dtype=g_before.dtype)
                    d = (g_after - g_before).abs()
                    print(f"[ARM-SOFT] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
                          f"rho(min/mean/max)={_stat3(rho)} |Δg|(min/mean/max)={_stat3(d)}")
                else:
                    print(f"[ARM-SOFT] lid={l} a_min={a_min:.2f} strength={strength:.3f} "
                          f"rho(min/mean/max)={_stat3(rho)}")

        if resmix_nudger is not None:
            try: resmix_nudger.nudge(step_size=-0.05 * float(max(guard_w, drv_weight)))
            except Exception: pass

        qv_state['armed_live'] = True
        qv_state['val_acc0']   = val_acc
        qv_state['rb_streak']  = 0

    # 6) 回退判定
    if qv_state['val_acc_last'] is None:
        qv_state['val_acc_last'] = val_acc

    if qv_state['armed_live']:
        improved = (val_acc >= qv_state['val_acc0'] - float(args.stagnation_tol))
        qv_state['rb_streak'] = 0 if improved else (qv_state['rb_streak'] + 1)

        if qv_state['rb_streak'] >= int(args.rollback_patience):
            gamma = float(getattr(args, 'rollback_gamma', 0.6))
            to_snapshot = (getattr(args, 'rollback_to', 'one') == 'snapshot')
            g_snap = qv_state.get('g_snap', {})

            for lid, wrapper in enumerate(pca_wrappers):
                # 当前 g
                if hasattr(wrapper, 'get_component_gains') and (wrapper.get_component_gains() is not None):
                    g_now = wrapper.get_component_gains().detach()
                elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
                    g_now = wrapper.comp_gain.detach()
                else:
                    continue

                # 目标
                if to_snapshot and (lid in g_snap):
                    tgt = g_snap[lid].to(g_now.device, dtype=g_now.dtype)
                else:
                    tgt = torch.ones_like(g_now)

                g_relax = g_now + gamma * (tgt - g_now)

                if hasattr(wrapper, 'set_component_gains'):
                    wrapper.set_component_gains(g_relax)
                elif hasattr(wrapper, 'comp_gain') and (wrapper.comp_gain is not None):
                    with torch.no_grad():
                        wrapper.comp_gain.copy_(g_relax.to(wrapper.comp_gain.device, dtype=wrapper.comp_gain.dtype))

                d = (g_relax - g_now).abs()
                print(f"[PCA-ROLLBACK] lid={lid} mode={getattr(args,'rollback_to','one')} "
                      f"gamma={gamma:.2f} |Δg|(min/mean/max)={_stat3(d)} "
                      f"g_before(min/mean/max)={_stat3(g_now)} g_after(min/mean/max)={_stat3(g_relax)}")

            if resmix_nudger is not None:
                try: resmix_nudger.nudge(step_size=-0.2)
                except Exception: pass

            qv_state['rb_streak'] = 0
            qv_state['val_acc0']  = val_acc

    qv_state['val_acc_last'] = val_acc

# ---------------------------
# Main
# ---------------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',type=str,default='textcls', choices=['vision','textcls','tokcls'])
    p.add_argument('--dataset', type=str, default='',
                  choices=['sst2','mnist','c10','c100'],
                  help='Universal dataset; if "sst2" routes to text pipeline.')
    p.add_argument('--text_dataset',type=str,default='sst2', choices=['sst2','ag_news','trec6','conll2003'])
    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--max_len',type=int,default=128)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model',type=str,default='distillbert_hf_gate', choices=list(MODEL_MAP.keys()))
    p.add_argument('--num_labels',type=int,default=2)
    p.add_argument('--insert_layers',type=str,default='last')
    p.add_argument('--weight_lr',type=float,default=3e-4)
    p.add_argument('--adam_eps',type=float,default=1e-8)
    p.add_argument('--adam_wd',type=float,default=0.0)

    # modules
    p.add_argument('--gate_enable',action='store_true')
    p.add_argument('--resmix_enable',action='store_true')
    p.add_argument('--resmix_from_dca_lr',type=float,default=0.0)

    # filters / fft
    p.add_argument('--filter_backend',type=str,default='lp_fixed', choices=['darts','lp_fixed','none'])
    p.add_argument('--lp_k',type=int,default=5)
    p.add_argument('--gate_tau',type=float,default=1.0)
    p.add_argument('--feature_mask', action='store_true')
    p.add_argument('--ff_gamma',type=float,default=1.0)
    p.add_argument('--ff_amin',type=float,default=0.8)
    p.add_argument('--ff_apply_on',type=str,default='input', choices=['input','output'])
    p.add_argument('--ff_use_drv_gate',action='store_true')

    # DCA
    p.add_argument('--dca_enable',action='store_true')
    p.add_argument('--dca_mode', type=str, default='e_ctr_resmix', choices=['e_ctr_resmix'])
    p.add_argument('--dca_beta',type=float,default=1.0)
    p.add_argument('--dca_w_lr',type=float,default=0.0)

    # derivative gate
    p.add_argument('--dca_derivative_enable',action='store_true')
    p.add_argument('--dca_drv_mode',type=str,default='ema', choices=['ema','window'])
    p.add_argument('--dca_drv_lambda',type=float,default=0.2)
    p.add_argument('--dca_drv_kappa',type=float,default=5.0)

    # energy-guided
    p.add_argument('--energy_guided', action='store_true')
    p.add_argument('--energy_gamma', type=float, default=1.0)
    p.add_argument('--energy_eps', type=float, default=1e-8)

    # PCA denoise
    p.add_argument('--pca_enable',action='store_true')
    p.add_argument('--pca_k',type=int,default=32)
    p.add_argument('--pca_lambda',type=float,default=0.8)
    p.add_argument('--pca_amin',type=float,default=0.85)
    p.add_argument('--pca_patience',type=int,default=2)

    # quickval selector
    p.add_argument('--arm_pick', type=str, default='prob', choices=['prob','softmix','deterministic'])
    p.add_argument('--arm_temp', type=float, default=1.0)
    p.add_argument('--arm_weights', type=str, default='pca=1,fft=1,lp=1')
    p.add_argument('--arm_thr', type=float, default=0.2)
    p.add_argument('--arm_invpow', type=float, default=2.0)
    p.add_argument('--arm_eps', type=float, default=1e-6)

    # hard exploration inside PCA (optional)
    p.add_argument('--arm_hard', action='store_true')
    p.add_argument('--arm_hard_mode', type=str, default='bernoulli', choices=['bernoulli', 'topk'])
    p.add_argument('--arm_hard_topk', type=int, default=0)
    p.add_argument('--arm_minprob', type=float, default=0.05)
    p.add_argument('--arm_maxprob', type=float, default=0.95)
    p.add_argument('--arm_prob_scale', type=float, default=1.0)
    p.add_argument('--arm_scope', type=str, default='both', choices=['comp', 'res', 'both'])

    # logging & quickval
    p.add_argument('--log_every',type=int,default=50)
    p.add_argument('--quickval_every',type=int,default=400)
    p.add_argument('--arch_val_batches',type=int,default=2)

    # spec reg (optional)
    p.add_argument('--spec_reg_enable',action='store_true')
    p.add_argument('--spec_reg_lambda',type=float,default=1e-4)
    p.add_argument('--spec_reg_power',type=float,default=1.0)

    # io
    p.add_argument('--log_dir',type=str,default='logs')
    p.add_argument('--run_tag',type=str,default='run')
    p.add_argument('--save_dir',type=str,default='checkpoints')
    p.add_argument('--save_name',type=str,default='model.pt')

    # SwanLab & CSV
    p.add_argument('--swan_enable', action='store_true')
    p.add_argument('--swan_project', type=str, default='default')
    p.add_argument('--swan_experiment', type=str, default='')
    p.add_argument('--swan_dir', type=str, default='swan_logs')
    p.add_argument('--csv_log', action='store_true')
    p.add_argument('--csv_path', type=str, default='optimization/log/metrics.csv')

    # Strict base
    p.add_argument('--strict_base', action='store_true',
                   help='Disable ALL optional inserts (gates/filters/PCA/ResMix) for a clean base.')

    # quickval/rollback configs
    p.add_argument('--trigger_train_acc', type=float, default=0.90)
    p.add_argument('--drift_patience', type=int, default=2)
    p.add_argument('--rollback_patience', type=int, default=2)
    p.add_argument('--stagnation_tol', type=float, default=1e-4)
    p.add_argument('--rollback_gamma', type=float, default=0.6)
    p.add_argument('--rollback_to', type=str, default='one', choices=['one','snapshot'])
    p.add_argument('--trigger_mode', type=str, default='worsen', choices=['worsen','stagnate','gap'])
    p.add_argument('--stagnation_patience', type=int, default=None)
    p.add_argument('--gap_thr', type=float, default=0.10)

    # split controls
    p.add_argument('--val_split', type=float, default=0.1)

    # dataloader runtime（可调）
    p.add_argument('--num_workers', type=int, default=0, help='train loader workers (0 for text, 4 for vision suggested)')
    p.add_argument('--pin_memory', action='store_true', help='pin memory for CUDA input')

    # NEW: val shuffle control (default True, reproducible with dedicated generator)
    p.add_argument('--val_shuffle', dest='val_shuffle', action='store_true',
                   help='Shuffle validation set every epoch with a reproducible generator.')
    p.add_argument('--no_val_shuffle', dest='val_shuffle', action='store_false')
    p.set_defaults(val_shuffle=True)

    args=p.parse_args()

    #
    set_seed(args.seed)
    enforce_cuda_determinism()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    # ---------- EMA state ----------
    class _EMA:
        def __init__(self, decay=0.9):
            self.decay = decay; self.v = None
        def update(self, x: float) -> float:
            x = float(x)
            self.v = x if self.v is None else (self.decay*self.v + (1-self.decay)*x)
            return self.v
    qv_state = {
        'train_acc_ema': _EMA(decay=0.9),
        'train_ce_ema':  _EMA(decay=0.9),
        'val_acc_last':  None, 'armed_live': False, 'val_acc0': None, 'rb_streak': 0,
        'best_val_acc':  None, 'stg_streak': 0,
    }

    # ---- dataset/task resolver & num_labels override ----
    ds = (args.dataset or '').strip().lower()
    if ds == 'sst2':
        args.task = 'textcls'; args.text_dataset = 'sst2'; args.num_labels = 2
    elif ds in ('mnist','c10','cifar10','c100','cifar100'):
        args.task = 'vision'
        args.num_labels = 10 if ds in ('mnist','c10','cifar10') else 100
        if ds == 'cifar10': args.dataset = 'c10'
        if ds == 'cifar100': args.dataset = 'c100'

    # --- logging backends ---
    global swan, csvlog
    if args.swan_enable:
        try:
            import swanlab
            swan = swanlab.init(
                project=args.swan_project,
                name=(args.swan_experiment or args.run_tag),
                work_dir=args.swan_dir,
                config=vars(args)
            )
            print("[SWAN] logging enabled.")
        except Exception as e:
            print(f"[SWAN] init failed: {e}")
            swan = None

    if args.csv_log:
        import csv
        os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
        class _CSVLogger:
            def __init__(self, path):
                self.path = path
                self._header_written = os.path.exists(path) and os.path.getsize(path) > 0
            def log(self, row: dict):
                hdr = list(row.keys())
                need_header = not self._header_written
                with open(self.path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=hdr)
                    if need_header:
                        w.writeheader(); self._header_written = True
                    w.writerow(row)
        csvlog = _CSVLogger(args.csv_path)
        print(f"[CSV] logging to {args.csv_path}")

    # ---- seed & device ----
    set_seed(args.seed)
    enforce_cuda_determinism()  # << 新增：关闭 TF32 与非确定性 SDPA/Flash
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- model ----
    ModelCls = MODEL_MAP[args.model]
    extra_kwargs = {}
    if args.task == 'textcls' and args.model in ('small_bert','small_transformer','distillbert_hf_gate'):
        extra_kwargs['text_task'] = 'cls'
        extra_kwargs['text_num_classes'] = args.num_labels
    if args.task == 'vision':
        import inspect
        if 'num_classes' in inspect.signature(ModelCls.__init__).parameters:
            extra_kwargs['num_classes'] = args.num_labels

    model = ModelCls(
        insert_layers=args.insert_layers,
        filter_backend=args.filter_backend, lp_k=args.lp_k,
        gate_tau=args.gate_tau,
        use_gate=args.gate_enable, use_resmix=args.resmix_enable,
        pca_enable=args.pca_enable, pca_k=args.pca_k, pca_amin=args.pca_amin,
        feature_mask=args.feature_mask,
        ff_gamma=args.ff_gamma, ff_amin=args.ff_amin,
        ff_apply_on=args.ff_apply_on, ff_use_drv_gate=args.ff_use_drv_gate,
        **extra_kwargs
    ).to(device)

    if args.strict_base and hasattr(model, 'disable_all_inserts'):
        print("[INFO] Strict base ON: disabling all inserts/gates/filters/PCA/ResMix.")
        model.disable_all_inserts()
        args.quickval_every = 0
        args.arch_val_batches = 0

    if args.task == 'vision':
        replaced = False
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
            in_f = model.classifier.in_features; model.classifier = nn.Linear(in_f, args.num_labels).to(device); replaced=True
        elif hasattr(model, 'head') and hasattr(model.head, 'in_features'):
            in_f = model.head.in_features; model.head = nn.Linear(in_f, args.num_labels).to(device); replaced=True
        elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
            in_f = model.fc.in_features; model.fc = nn.Linear(in_f, args.num_labels).to(device); replaced=True
        if replaced:
            print(f"[FIX] force reset classifier head -> num_classes={args.num_labels}")

    # ---- optim ----
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.weight_lr, eps=args.adam_eps, weight_decay=args.adam_wd)

    # ---- DCA/derivative/resmix/pca/guard ----
    dca=None
    if args.dca_enable and (DCA is not None):
        if DCACfg is not None: dca = DCA(cfg=DCACfg(enable=True), num_classes=args.num_labels)
        else:
            try: dca = DCA(num_classes=args.num_labels)
            except TypeError:
                try: dca = DCA()
                except Exception as e: print(f"[WARN] DCA construction failed: {e}"); dca=None
        if dca is not None and hasattr(dca, "attach_hooks"):
            if not getattr(dca, "_attached", False):
                dca.attach_hooks(model); dca._attached=True; print("[DCA] hooks attached")

    dca_drv=None
    if args.dca_derivative_enable and (DCADerivative is not None):
        try: dca_drv = DCADerivative()
        except Exception as e: print(f"[WARN] DCADerivative() failed, disable derivative gate: {e}")

    resmix_nudger = SoftResMixNudger(model) if args.resmix_enable and args.resmix_from_dca_lr>0 else None

    pca_mgr=None
    if args.pca_enable and (not args.strict_base):
        pca_mgr = PCAOverlapManager(k=args.pca_k, ema_decay=0.99, device=str(device))
        if hasattr(model, 'attach_pca_manager'):
            model.attach_pca_manager(pca_mgr)

    guard=OverfitGuard(OverfitGuardConfig(patience=args.pca_patience,
                                          kappa=args.dca_drv_kappa, clip=1.0))

    # ====== [B] Generators ======
    gen_train = make_global_train_generator(args.seed)               # 训练复用
    gen_val   = make_global_train_generator(args.seed + 1) if args.val_shuffle else None  # 验证可选

    # ---- loaders ----
    collate = getattr(model, 'collate_fn', None)

    if args.task == 'textcls':
        if (args.dataset == 'sst2') or (args.text_dataset == 'sst2'):
            train_loader, val_loader, test_loader = build_sst2_loaders(
                args.batch, collate_fn=collate, val_split=args.val_split, seed=args.seed,
                num_workers=args.num_workers, pin_memory=args.pin_memory,
                train_generator=gen_train,
                val_shuffle=args.val_shuffle, val_generator=gen_val
            )
        else:
            raise ValueError(f"Only 'sst2' is implemented for textcls.")

    elif args.task == 'vision':
        if args.dataset not in ('mnist','c10','c100'):
            raise ValueError("When --task vision, set --dataset in ['mnist','c10','c100'].")
        clsname = model.__class__.__name__.lower()
        if hasattr(model, 'bert') or 'bert' in clsname or 'distill' in clsname:
            print(f"[ERROR] Text model cannot run on VISION dataset '{args.dataset}'.")
            sys.exit(2)

        train_loader, val_loader, test_loader, _ = build_vision_loaders(
            name=args.dataset, batch=args.batch, root=args.data_root,
            val_split=args.val_split, seed=args.seed,
            num_workers=(args.num_workers or 4), pin_memory=True,
            train_generator=gen_train,
            val_shuffle=args.val_shuffle, val_generator=gen_val
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # --- train loop ---
    state={'step':0,'best_acc':-1,'last_qv_step':0}
    for epoch in range(1,args.epochs+1):
        m=run_epoch(model,train_loader,opt,device,state,args,train=True,pca_mgr=pca_mgr)
        print(f"[E{epoch:02d}] train loss={m['loss']:.4f} acc={m['acc']:.4f}")

        # EMA proxies
        qv_state['train_acc_ema'].update(float(m['acc']))
        qv_state['train_ce_ema'].update(float(m.get('loss', 1.0 - m['acc'])))

        if swan is not None:
            swan.log({"train/epoch_loss": float(m['loss']), "train/epoch_acc": float(m['acc'])}, step=epoch)
        if csvlog is not None:
            csvlog.log({"stage":"train_epoch","epoch":epoch,
                        "loss":round(float(m['loss']),6),"acc":round(float(m['acc']),6)})

        if test_loader is not None:
            tm=evaluate(model,test_loader,device,args)
            print(f"[E{epoch:02d}]  test loss={tm['loss']:.4f} acc={tm['acc']:.4f}")
            if swan is not None:
                swan.log({"test/epoch_loss": float(tm['loss']), "test/epoch_acc": float(tm['acc'])}, step=epoch)
            if csvlog is not None:
                csvlog.log({"stage":"test_epoch","epoch":epoch,
                            "loss":round(float(tm['loss']),6),"acc":round(float(tm['acc']),6)})

        # 差值触发 quickval
        if (args.quickval_every>0) and (not args.strict_base):
            if state['step'] - state.get('last_qv_step',0) >= args.quickval_every:
                opt.zero_grad(set_to_none=True)
                print(f"[QVAL] step={state['step']} (Δ={state['step']-state.get('last_qv_step',0)}) epoch={epoch} -> quickval")
                quickval_arch_step(model,val_loader,device,args,dca,dca_drv,resmix_nudger,pca_mgr,guard,qv_state)
                state['last_qv_step'] = state['step']

        vm=evaluate(model,val_loader,device,args)
        print(f"[E{epoch:02d}] valid loss={vm['loss']:.4f} acc={vm['acc']:.4f}")
        if swan is not None:
            swan.log({"valid/epoch_loss": float(vm['loss']), "valid/epoch_acc": float(vm['acc'])}, step=epoch)
        if csvlog is not None:
            csvlog.log({"stage":"valid_epoch","epoch":epoch,
                        "loss":round(float(vm['loss']),6),"acc":round(float(vm['acc']),6)})

        if vm['acc']>state['best_acc']:
            state['best_acc']=vm['acc']
            os.makedirs(args.save_dir,exist_ok=True)
            save_path=os.path.join(args.save_dir,args.save_name)
            torch.save({'model':model.state_dict(),'args':vars(args),'epoch':epoch}, save_path)
            print(f"[E{epoch:02d}] saved best to {save_path}")

if __name__=='__main__':
    main()
