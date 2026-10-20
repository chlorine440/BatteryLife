"""Resolve public CLI configurations without executing their shell scripts."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import shlex
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2021, 42, 2024)
MODELS = ("CPTransformer", "CPMLP", "CPGRU", "CPBiGRU", "CPLSTM", "CPBiLSTM",
          "MLP", "CNN", "DLinear", "PatchTST", "Autoformer", "iTransformer", "MICN")
RNN_MODELS = frozenset(("CPGRU", "CPLSTM", "CPBiGRU", "CPBiLSTM"))
TABLE_MODELS = frozenset(("CPTransformer", "CPMLP", "CPGRU", "CPLSTM"))
COMMON = dict(dataset="MIX_large", data="Dataset_original", task_name="classification",
              is_training=1, seq_len=1, early_cycle_threshold=100, charge_discharge_length=300,
              train_epochs=100, least_epochs=5, patience=5, num_workers=32,
              loss="MSE", lradj="constant", use_amp=False, weighted_loss=False,
              weighted_sampling=False, wd=0.0, itr=1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parser_specs(root: Path = ROOT) -> dict:
    """AST only: importing run_main would launch training."""
    specs = {}
    for node in ast.walk(ast.parse((root / "run_main.py").read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "parser"
                and node.func.attr == "add_argument"):
            continue
        key = ast.literal_eval(node.args[0]).removeprefix("--")
        kw = {v.arg: v.value for v in node.keywords}
        typ = kw.get("type")
        action = ast.literal_eval(kw["action"]) if "action" in kw else None
        default = ast.literal_eval(kw["default"]) if "default" in kw else (
            False if action == "store_true" else None)
        specs[key] = {"type": typ.id if isinstance(typ, ast.Name) else "str",
                      "action": action, "default": default}
    return specs


def script_args(model: str, root: Path = ROOT) -> dict:
    if model not in MODELS:
        raise ValueError(f"not an allowed benchmark model: {model}")
    text = (root / "train_eval_scripts" / f"{model}.sh").read_text()
    if text.count("run_main.py") != 1:
        raise ValueError(f"expected a single training invocation in {model}.sh")
    before, after = text.split("run_main.py")
    variables = {}
    for line in before.splitlines():
        parts = shlex.split(line, comments=True)
        if len(parts) == 1 and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", parts[0]):
            k, v = parts[0].split("=", 1)
            variables[k] = v
    tokens = shlex.split(after.replace("\\\n", " "), comments=True)
    specs, result = parser_specs(root), {}
    pos = 0
    while pos < len(tokens):
        key = tokens[pos].removeprefix("--")
        if not tokens[pos].startswith("--") or key not in specs:
            raise ValueError(f"unsupported shell CLI token: {tokens[pos]}")
        pos += 1
        if specs[key]["action"] == "store_true":
            result[key] = True
            continue
        value = tokens[pos]
        pos += 1
        def replace(match):
            name = match.group(1) or match.group(2)
            if name not in variables:
                raise ValueError(f"undefined script variable: {name}")
            return variables[name]
        value = re.sub(r"\$\{(\w+)\}|\$(\w+)", replace, value)
        if "$" in value or "`" in value:
            raise ValueError("shell expressions are not evaluated")
        result[key] = {"int": int, "float": float, "str": str}[specs[key]["type"]](value)
    return result


def resolve_config(model: str, root: Path = ROOT) -> dict:
    args = {k: v["default"] for k, v in parser_specs(root).items()}
    args.update(script_args(model, root))
    if model in TABLE_MODELS:
        rows = []
        for line in (root / "assets/Selected_hyperparameters.md").read_text().splitlines():
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) == 10 and cols[:2] == [model, "Li-ion"]:
                rows.append(dict(zip(("batch_size", "d_model", "d_ff", "e_layers", "d_layers",
                                      "dropout", "learning_rate"),
                                     [float(c) if i >= 5 else int(c) for i, c in enumerate(cols[3:])])) )
        if len(rows) != 3 or any(r != rows[0] for r in rows):
            raise ValueError(f"paper table needs three identical Li-ion configs: {model}")
        args.update(rows[0])
    args.update(COMMON, model=model, model_id=model)
    # Paths and seeds are supplied by the suite, never by placeholder shell paths.
    for key in ("seed", "root_path", "checkpoints", "model_comment"):
        args.pop(key)
    return args


def provenance(model: str, root: Path = ROOT) -> dict:
    names = ["run_main.py", f"train_eval_scripts/{model}.sh"]
    if model in TABLE_MODELS:
        names.append("assets/Selected_hyperparameters.md")
    return {"parameter_source": "paper_Li-ion_table" if model in TABLE_MODELS else "repository_script",
            "fallback": "original run_main parser defaults",
            "sha256": {n: sha256(root / n) for n in names}}


def instantiate(args: dict):
    model = args["model"]
    if model not in MODELS:
        raise ValueError(f"not an allowed benchmark model: {model}")
    return importlib.import_module(f"models.{model}").Model(Namespace(**args)).float()


def cli_args(args: dict) -> list[str]:
    specs = parser_specs()
    result = []
    for key, value in args.items():
        if specs[key]["action"] == "store_true":
            if value:
                result.append(f"--{key}")
        elif value is not None:
            result.extend((f"--{key}", str(value)))
    return result
