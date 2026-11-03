"""Wikitext-2-raw, tokenised with bert-base-uncased WordPiece (as in the paper),
cached as one token stream per split plus, for every token, the index of the
Wikipedia article it comes from (the clusters for the bootstrap in analyze.py).

    python -m grassman.data --data-dir data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

SPLITS = ("train", "validation", "test")
TOKENIZER = "bert-base-uncased"

# A level-1 heading " = Title = " starts a new article; " = = Section = = " does not.
_ARTICLE = re.compile(r"^= [^=].* =$")


def is_article_start(line: str) -> bool:
    return bool(_ARTICLE.match(line.strip()))


def article_ids(lines: list[str]) -> list[int]:
    """Article index of each line (0 for text before the first heading)."""
    ids, current = [], -1
    for line in lines:
        if is_article_start(line):
            current += 1
        ids.append(max(current, 0))
    return ids


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
        docs = article_ids(lines)
        out[split] = torch.tensor([t for line in ids for t in line], dtype=torch.long)
        doc = torch.tensor([d for line, d in zip(ids, docs) for _ in line], dtype=torch.long)
        torch.save(out[split], data_dir / f"{split}.pt")
        torch.save(doc, data_dir / f"{split}_doc.pt")
    return out


def _cached(data_dir: Path) -> bool:
    return all((data_dir / f"{s}{suffix}.pt").exists() for s in SPLITS for suffix in ("", "_doc"))


def load(data_dir: str | Path) -> dict[str, torch.Tensor]:
    data_dir = Path(data_dir)
    if not _cached(data_dir):
        return build(data_dir)
    return {s: torch.load(data_dir / f"{s}.pt", weights_only=True) for s in SPLITS}


def load_docs(data_dir: str | Path) -> dict[str, torch.Tensor]:
    """Article index for every token of each split (same length as load())."""
    data_dir = Path(data_dir)
    if not _cached(data_dir):
        build(data_dir)
    return {s: torch.load(data_dir / f"{s}_doc.pt", weights_only=True) for s in SPLITS}


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
    docs = load_docs(args.data_dir)
    for split, t in load(args.data_dir).items():
        print(f"{split:>10}: {len(t):,} tokens, {int(docs[split].max()) + 1} articles")
