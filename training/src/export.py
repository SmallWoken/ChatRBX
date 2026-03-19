"""
Export a trained checkpoint to Roblox-loadable weight files.

Usage (from training/):
    python src/export.py

Outputs:
    training/weights/weights.json   — raw JSON (useful for debugging)
    training/weights/weights.lua    — Lua table, paste into a Roblox ModuleScript

The Lua format looks like:
    return {
        config  = { vocab_size=67, n_embd=32, ... },
        chars   = "abcde...",   -- itos: index (1-based) → character
        weights = {
            ["wte.weight"] = { {f, f, f, ...}, ... },  -- 2-D tensors as nested arrays
            ["ln_f.bias"]  = { f, f, f, ... },          -- 1-D tensors as flat arrays
            ...
        },
    }

Note on size: a default ~30K-param model exports to roughly 240 KB as JSON and
~280 KB as Lua text — small enough to paste into a single ModuleScript.
"""

import json
import sys
import torch
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
CKPT_PATH   = WEIGHTS_DIR / "checkpoint.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_list(t: torch.Tensor) -> list:
    """Convert a tensor to a nested Python list, rounded to 6 sig-figs."""
    arr = t.detach().float().cpu().numpy()
    if arr.ndim == 1:
        return [round(float(v), 6) for v in arr]
    elif arr.ndim == 2:
        return [[round(float(v), 6) for v in row] for row in arr]
    elif arr.ndim == 3:
        return [[[round(float(v), 6) for v in col] for col in row] for row in arr]
    else:
        raise ValueError(f"Unsupported tensor ndim: {arr.ndim}")


def fmt_float(v: float) -> str:
    """Compact float string — no trailing zeros, max 6 sig-figs."""
    s = f"{v:.6g}"
    return s


def lua_value(obj, indent: int = 0) -> str:
    """Recursively render a Python value as Lua source."""
    pad  = "\t" * indent
    pad1 = "\t" * (indent + 1)

    if isinstance(obj, float) or isinstance(obj, int):
        return fmt_float(float(obj))
    elif isinstance(obj, str):
        # Escape special chars for a Lua long string isn't worth it; use quotes.
        escaped = obj.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\0", "\\0")
        return f'"{escaped}"'
    elif isinstance(obj, list):
        if not obj:
            return "{}"
        # Flat list of numbers → single line if short enough
        if isinstance(obj[0], (int, float)):
            inner = ", ".join(fmt_float(float(v)) for v in obj)
            line  = "{" + inner + "}"
            if len(line) < 120:
                return line
            # Too long — one number per line
            lines = [pad1 + fmt_float(float(v)) for v in obj]
            return "{\n" + ",\n".join(lines) + "\n" + pad + "}"
        else:
            # List of lists (2D+ tensor rows)
            rows = []
            for item in obj:
                rows.append(pad1 + lua_value(item, indent + 1))
            return "{\n" + ",\n".join(rows) + "\n" + pad + "}"
    elif isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            key = f'["{k}"]' if not k.isidentifier() else k
            lines.append(f"{pad1}{key} = {lua_value(v, indent + 1)}")
        return "{\n" + ",\n".join(lines) + "\n" + pad + "}"
    else:
        raise TypeError(f"Cannot convert {type(obj)} to Lua")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not CKPT_PATH.exists():
        sys.exit(f"No checkpoint found at {CKPT_PATH}.\nRun  python src/train.py  first.")

    print(f"Loading checkpoint from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)

    config  = ckpt["config"]
    vocab   = ckpt["vocab"]
    state   = ckpt["model_state"]

    # Build export dict: name → nested list
    weights_dict: dict[str, list] = {}
    for name, tensor in state.items():
        # Skip the causal mask (not needed at inference time — we recompute it)
        if name.endswith(".bias") and tensor.dim() == 4:
            print(f"  Skipping causal mask: {name}")
            continue
        weights_dict[name] = tensor_to_list(tensor)
        shape_str = "x".join(str(s) for s in tensor.shape)
        print(f"  Exported {name:50s} {shape_str}")

    # chars string: index i (0-based Python) = chars[i]
    chars_str = "".join(vocab["chars"])

    export = {
        "config":  config,
        "chars":   chars_str,
        "weights": weights_dict,
    }

    # --- JSON ---
    json_path = WEIGHTS_DIR / "weights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, separators=(",", ":"))
    print(f"\nJSON  → {json_path}  ({json_path.stat().st_size // 1024} KB)")

    # --- Lua ---
    lua_path = WEIGHTS_DIR / "weights.lua"
    with open(lua_path, "w", encoding="utf-8") as f:
        f.write("-- Auto-generated by export.py — do not edit manually\n")
        f.write("-- Paste this file's contents into a Roblox ModuleScript\n\n")
        f.write("return ")
        f.write(lua_value(export))
        f.write("\n")
    print(f"Lua   → {lua_path}  ({lua_path.stat().st_size // 1024} KB)")

    print("\nDone! Paste weights.lua into a ModuleScript in ReplicatedStorage.")
    print("Lua index note: weights are 1-indexed in Lua — add 1 when looking up chars.")


if __name__ == "__main__":
    main()
