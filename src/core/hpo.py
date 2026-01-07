import os
import pandas as pd
from copy import deepcopy
from src.core.trainer import train_one
from src.utils.io import ensure_dir

def lr_sweep(cfg_base, model_builder, loaders, device, exp_name, out_csv):
    train_loader, val_loader, test_loader = loaders
    rows=[]
    best=None
    best_lr=None
    for lr in cfg_base["lr_list"]:
        cfg = deepcopy(cfg_base)
        cfg["lr"]=float(lr)
        cfg["epochs"]=int(cfg_base["epochs_hpo"])
        model = model_builder()
        val_acc, test_acc = train_one(cfg, model, train_loader, val_loader, test_loader,
                                      device=device, exp_name=f"{exp_name}_lr{lr}",
                                      out_csv_path=os.path.join(os.path.dirname(out_csv), f"curve_lr{lr}.csv"),
                                      router_mode=cfg.get("router_mode", False),
                                      alt_mode=cfg.get("alt_mode","none"))
        rows.append({"lr":lr,"val_acc":val_acc,"test_acc":test_acc})
        if best is None or val_acc>best:
            best=val_acc
            best_lr=lr
    df=pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return float(best_lr)
