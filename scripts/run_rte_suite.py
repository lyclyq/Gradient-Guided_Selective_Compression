import argparse
from src.experiments.rte_suite import run_suite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--seeds", type=str, default="2")
    ap.add_argument("--run_tag", type=str, default="s1")
    ap.add_argument("--config", type=str, default="configs/rte_default.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_suite(model_name=args.model_name, seeds=seeds, run_tag=args.run_tag, config_path=args.config)

if __name__ == "__main__":
    main()
