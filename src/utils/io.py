import os

def ensure_dir(p:str):
    os.makedirs(p, exist_ok=True)

def build_output_paths(root, model_name, task, exp, run_tag):
    base = os.path.join(root, f"{model_name}_{task}", exp, run_tag)
    return {
        "base": base,
        "hpo": os.path.join(base, "hpo"),
        "final": os.path.join(base, "final"),
        "plots": os.path.join(base, "plots"),
        "log": os.path.join(base, "log"),
    }
