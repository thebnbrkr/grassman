"""Wikitext-2-raw, tokenised with bert-base-uncased WordPiece (as in the paper),
cached as one token stream per split.

    python -m grassman.data --data-dir data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

SPLITS = ("train", "validation", "test")
TOKENIZER = "bert-base-uncased"


def build(data_dir: str | Path) -> dict[str, torch.Tensor]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    out = {}
    for split in SPLITS:
        lines = [line for line in ds[split]["text"] if line.strip()]
        ids = tok(lines, add_special_tokens=False)["input_ids"]
        out[split] = torch.tensor([t for line in ids for t in line], dtype=torch.long)
        torch.save(out[split], data_dir / f"{split}.pt")
    return out


def load(data_dir: str | Path) -> dict[str, torch.Tensor]:
    data_dir = Path(data_dir)
    if not all((data_dir / f"{s}.pt").exists() for s in SPLITS):
        return build(data_dir)
    return {s: torch.load(data_dir / f"{s}.pt", weights_only=True) for s in SPLITS}


def blocks(stream: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a token stream into non-overlapping (x, y) blocks, y = x shifted by
    one. The final partial block is dropped."""
    n = (len(stream) - 1) // block_size
    x = stream[: n * block_size].view(n, block_size)
    y = stream[1 : n * block_size + 1].view(n, block_size)
    return x, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    for split, t in load(args.data_dir).items():
        print(f"{split:>10}: {len(t):,} tokens")
