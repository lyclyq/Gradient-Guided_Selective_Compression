#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}

IGNORE_SUFFIX = {
    ".pyc",
    ".pyo",
    ".swp",
}

def dump_tree(
    root: Path,
    out_txt: Path,
    max_depth: Optional[int] = None,
):
    root = root.resolve()
    lines = []

    def walk(p: Path, depth: int):
        if max_depth is not None and depth > max_depth:
            return

        indent = "  " * depth
        lines.append(f"{indent}{p.name}/")

        try:
            items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            lines.append(f"{indent}  [PermissionError]")
            return

        for it in items:
            if it.is_dir():
                if it.name in IGNORE_DIRS:
                    continue
                walk(it, depth + 1)
            else:
                if it.suffix in IGNORE_SUFFIX:
                    continue
                lines.append(f"{indent}  {it.name}")

    walk(root, 0)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Project tree written to: {out_txt}")
    print(f"[INFO] Root: {root}")
    print(f"[INFO] Lines: {len(lines)}")


if __name__ == "__main__":
    # ====== 你只需要改这两行（默认已按你项目路径填好） ======
    ROOT_DIR = Path("/home/lyclyq/Optimization/grad-shake-align")
    OUTPUT_TXT = Path("/home/lyclyq/Optimization/grad-shake-align/project_tree.txt")

    # max_depth=None 表示不限制深度；比如设 4 就最多展开 4 层
    dump_tree(ROOT_DIR, OUTPUT_TXT, max_depth=None)
