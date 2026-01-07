import os, json, logging
from datetime import datetime
from copy import deepcopy
from src.utils.seed import set_seed
from src.utils.io import build_output_paths, ensure_dir
from src.data.glue_rte import get_loaders
from src.model.build import build_model, inject_adapters
from src.core.hpo import lr_sweep
from src.core.trainer import train_one, evaluate
from src.core.plotting import plot_compare

OUTPUT_ROOT = "outputs"

def _device_from_cfg(cfg):
    import torch
    dev = cfg.get("device","cuda")
    if dev == "cuda" and (not torch.cuda.is_available()):
        return "cpu"
    return dev

def run_suite(model_name:str, seeds, run_tag:str, config_path:str):
    with open(config_path,"r") as f:
        base_cfg=json.load(f)
    task="rte"
    for seed in seeds:
        cfg0=deepcopy(base_cfg)
        cfg0["seed"]=seed
        set_seed(seed)
        train_loader, val_loader, test_loader = get_loaders(model_name, cfg0["batch_size"], cfg0["max_length"], seed)
        loaders=(train_loader, val_loader, test_loader)
        device=_device_from_cfg(cfg0)

        exps = [
            ("ab_only", dict(router_mode=False, alt_mode="ab_only", use_alt=False)),
            ("offline_project", dict(router_mode=False, alt_mode="alt_only", use_alt=True)),
            ("online_nodistill", dict(router_mode=False, alt_mode="none", use_alt=True)),
            ("ours_soft_absorb", dict(router_mode=True, alt_mode="none", use_alt=True)),
        ]

        curves=[]
        for exp, meta in exps:
            paths = build_output_paths(OUTPUT_ROOT, model_name, task, exp, run_tag)
            for k in ("hpo","final","plots","log"):
                ensure_dir(paths[k])
            log_name = f"{exp}_seed{seed}_{run_tag}.log"
            log_path = os.path.join(paths["log"], log_name)
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # HPO
            cfg=deepcopy(cfg0)
            cfg.update(meta)
            logger.info("Experiment start: %s seed=%s run_tag=%s", exp, seed, run_tag)
            logger.info("Config: %s", json.dumps(cfg, sort_keys=True))
            def model_builder():
                m = build_model(model_name, num_labels=2)
                m = inject_adapters(m, r=cfg["r"], r_alt=cfg["r_alt"], use_alt=meta["use_alt"], ab_only=(exp=="ab_only"))
                return m

            hpo_csv=os.path.join(paths["hpo"], "lr_sweep.csv")
            best_lr = lr_sweep(cfg, model_builder, loaders, device, exp, hpo_csv)
            logger.info("Best lr selected: %s", best_lr)

            # Train final
            cfg_f=deepcopy(cfg)
            cfg_f["lr"]=best_lr
            cfg_f["epochs"]=int(cfg0["epochs_train"])
            curve_csv=os.path.join(paths["final"], "curve.csv")
            model = model_builder()
            val_acc, test_acc = train_one(cfg_f, model, train_loader, val_loader, test_loader,
                                          device=device, exp_name=exp, out_csv_path=curve_csv,
                                          router_mode=cfg_f.get("router_mode",False),
                                          alt_mode=cfg_f.get("alt_mode","none"),
                                          logger=logger)
            logger.info(
                "Experiment end: %s val_acc=%.4f test_acc=%.4f end_time=%s",
                exp,
                val_acc,
                test_acc,
                datetime.now().isoformat(timespec="seconds"),
            )
            logger.removeHandler(file_handler)
            file_handler.close()

            # Offline projection step for offline_project baseline:
            if exp == "offline_project":
                # absorb all mature alt into AB at end and evaluate AB-only
                from src.core.trainer import _collect_adapters
                adapters=_collect_adapters(model)
                for a in adapters:
                    if a.use_alt and a.use_ab:
                        # AB exists in this experiment; but AB grads were zero. Still initialized. Absorb aggressively.
                        a.absorb_alt_into_ab(r=cfg_f["r"], alpha=1.0)
                        # remove alt effect by zeroing z
                        a.z.data.zero_()
                val2 = evaluate(model, val_loader, device)
                test2 = evaluate(model, test_loader, device)
                # append note
                import pandas as pd
                df=pd.read_csv(curve_csv)
                df.loc[len(df)] = {"step":df["step"].max() if len(df)>0 else 0, "epoch":cfg_f["epochs"],
                                   "loss":None, "val_acc":val2, "test_acc":test2}
                df.to_csv(curve_csv, index=False)

            curves.append((exp, curve_csv))

        # plot compare
        plot_path=os.path.join(OUTPUT_ROOT, f"{model_name}_{task}", "plots")
        ensure_dir(plot_path)
        out_png=os.path.join(plot_path, f"compare_seed{seed}_{run_tag}.png")
        plot_compare(curves, out_png, title=f"{model_name} {task} seed {seed}")
