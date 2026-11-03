"""Write the Colab notebooks for the follow-up tests.

    python notebooks/make_notebooks.py

speed.ipynb            fast vs paper-order Grassmann: same numbers? how much faster? where does time go?
choose_vs_reach.ipynb  is the gap to attention about choosing which token, or about reach?
hybrid.ipynb           how much of the gap does one attention layer recover?
readouts.ipynb         are the edge choices readable, used, and meaningful?

All are validation-only (--no-test) and save results under /content/results.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def setup(runs_file, data=True):
    cells = [
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code(f"""
import os, json, subprocess, sys
REPO, DATA, OUTDIR = '/content/grassman', '/content/data', '/content/results'
RUNS = f'{{OUTDIR}}/{runs_file}'
os.makedirs(OUTDIR, exist_ok=True)
if os.path.exists(REPO):
    !git -C {{REPO}} pull -q
else:
    !git clone -q https://github.com/thebnbrkr/grassman.git {{REPO}}
%cd {{REPO}}
!git log --oneline -1
!pip install -q datasets transformers
"""),
        md("Optional: upload an earlier `results.zip` so finished runs are skipped (cancel to start fresh)."),
        code("""
from google.colab import files
up = files.upload()
for name in up:
    !cd /content && unzip -o -q "{name}"
!ls -la {OUTDIR}
"""),
        md("Unit tests (about a minute). Stop if anything fails."),
        code("!python -m pytest -q"),
    ]
    if data:
        cells.append(code("!python -m grassman.data --data-dir {DATA}"))
    return cells


TRAIN_HELPER = code("""
def train(mixer, seed, epochs=5, lr=1e-3, attn='', save=None, tag='pilot'):
    cmd = [sys.executable, '-m', 'grassman.train', '--mixer', mixer, '--seed', str(seed), '--lr', str(lr),
           '--epochs', str(epochs), '--tag', tag, '--data-dir', DATA, '--out', RUNS,
           '--skip-existing', '--no-test']
    if attn:
        cmd += ['--attn-layers', attn]
    if save:
        cmd += ['--save', save]
    print('>>', mixer, f'attn@{attn}' if attn else '', 'seed', seed, flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
""")

CURVES = code("""
import matplotlib.pyplot as plt
from grassman import analyze
rows = analyze.load(RUNS)
fig, ax = plt.subplots(figsize=(8, 4.5))
for r in rows:
    h = r['history']
    ax.plot([e['epoch'] for e in h], [e['val_ppl'] for e in h], ls='-' if r['seed'] == 0 else '--',
            label=f"{analyze.name(r)} s{r['seed']}")
ax.set_xlabel('epoch'); ax.set_ylabel('validation perplexity'); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
plt.savefig(RUNS.replace('.jsonl', '_curves.png'), dpi=150, bbox_inches='tight'); plt.show()
""")

DOWNLOAD = [
    md("Download the results before the runtime is deleted."),
    code("""
!cd /content && zip -qr results.zip results
from google.colab import files
files.download('/content/results.zip')
"""),
]


def speed():
    return [md("""
# Speed: the fast Grassmann layer

The paper projects each offset's 496 Plücker numbers to 256 and then averages the six results. Projecting and averaging are both linear, so this notebook's code averages first and projects once. It also builds the average from the 32×32 pair products, so the slow upper-triangle lookup happens once instead of six times, and it takes the normaliser from Lagrange's identity, ‖a∧b‖² = ‖a‖²‖b‖² − (a·b)². Same maths, fewer operations.

1. **Same numbers?** Fast path vs the paper's order on this GPU, in float32: outputs and every gradient.
2. **How much faster?** Training tokens/s for each model, fast vs paper order.
3. **Where does the time go?** A profile of a few training steps, to explain the shared slowdown on the L4.

No data download is needed: speed is measured on random tokens. About 5 minutes on an A100.
"""),
            *setup("speed.json", data=False),
            md("## 1. Same numbers?\n\nDifferences of about 1e-5 or less are float32 rounding; anything larger is a bug."),
            code("""
import torch
from grassman.bench import fast_vs_reference, throughput, set_reference
dev = torch.device('cuda')
for r in fast_vs_reference(dev):
    print(f"{r['kind']:<8} normalize={str(r['normalize']):<5}  max output diff {r['max_out_diff']:.2e}   "
          f"max gradient diff {r['max_grad_diff']:.2e}")
"""),
            md("## 2. How much faster? (training tokens/s, batch 32 × 128, bf16)"),
            code("""
from grassman.model import Config
results = {}
for m in ['grassmann', 'sym', 'outer', 'concat', 'concat32', 'g-select', 'g-dilated', 'g-both', 'g-both-angle',
          'hyena', 'swa', 'attention']:
    cfg = Config(mixer=m)
    fast = throughput(cfg, dev)
    ref = throughput(cfg, dev, reference=True) if m in ('grassmann', 'sym', 'outer', 'concat', 'concat32', 'g-dilated') else None
    results[m] = {'fast': fast, 'paper_order': ref}
    print(f"{m:<13} fast {fast:>10,.0f} tok/s" + (f"   paper order {ref:>10,.0f}   speed-up {fast / ref:4.1f}x" if ref else ''))
json.dump({'gpu': torch.cuda.get_device_name(), 'tok_per_s': results}, open(RUNS, 'w'), indent=1)
"""),
            md("## 3. Where does the time go?\n\nTop GPU operations over 10 training steps. If the same operations dominate for `attention` and `grassmann`, the cost is in the shared parts (embedding, output layer, loss, optimizer), not the mixer."),
            code("""
from torch.profiler import profile, ProfilerActivity
from grassman.model import LM
for m in ['attention', 'grassmann']:
    torch.manual_seed(0)
    model = LM(Config(mixer=m)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 30522, (32, 128), device=dev)
    def step():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            step()
        torch.cuda.synchronize()
    print(f'===== {m} =====')
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=15))
"""),
            *DOWNLOAD]


def choose_vs_reach():
    return [md("""
# Choose vs reach: why is Grassmann 26% behind attention?

The paper's layer averages six fixed offsets (1–16 tokens back) with equal weights. Attention instead **chooses** which earlier token to use, from the content, and **reaches** the whole block. This notebook separates the two with a 2×2 design on the paper's Plücker features:

| | equal-weight average | learned edge weights (graph attention over the offsets) |
|---|---|---|
| **offsets 1–16** (the paper's) | `grassmann` | `g-select` |
| **offsets 1, 2, 4, 16, 64, 127** (same count, whole block) | `g-dilated` | `g-both` |

The learned weights are a softmax over the 6 edges, scored from [z_t; z_{t−Δ}] (GATv2-style). The references are `swa` (chooses among the last 16) and `attention` (chooses among all).

2 seeds × 5 epochs at LR 1e-3, validation only. Pilot speeds before the speed fix: ~40–50 min on an A100.
"""),
            *setup("choose_vs_reach.jsonl"),
            TRAIN_HELPER,
            code("""
MIXERS = ['grassmann', 'g-select', 'g-dilated', 'g-both', 'swa', 'attention']
SEEDS = [0, 1]
for seed in SEEDS:
    for m in MIXERS:
        train(m, seed)
"""),
            md("## Results\n\nValidation perplexity, lower is better. 'vs grassmann' is per seed; negative means better than the paper's layer."),
            code("""
from grassman import analyze
rows = analyze.load(RUNS)
print(analyze.val_table(rows, 'grassmann'))

mean = lambda m: sum(r['best_val_ppl'] for r in rows if analyze.name(r) == m) / max(1, sum(analyze.name(r) == m for r in rows))
g, s, d, b = (mean(m) for m in ('grassmann', 'g-select', 'g-dilated', 'g-both'))
print(f\"\"\"
2x2 (mean validation perplexity)
                      equal weights   learned weights
  offsets 1-16        {g:13.1f}   {s:15.1f}
  offsets to 127      {d:13.1f}   {b:15.1f}

effect of choosing: {100 * ((s / g - 1) + (b / d - 1)) / 2:+.1f}%   (average over both reaches)
effect of reach:    {100 * ((d / g - 1) + (b / s - 1)) / 2:+.1f}%   (average over both weightings)
attention:          {mean('attention'):.1f}    swa: {mean('swa'):.1f}
gap to attention closed by g-both: {100 * (g - b) / (g - mean('attention')):.0f}%
\"\"\")
"""),
            CURVES,
            md("""
## How to read it

- **Choosing helps, reach doesn't:** the gap is about selecting the right token. `g-select` should approach `swa`.
- **Reach helps, choosing doesn't:** the gap is about seeing far back. That argues for long filters (Hyena) over local pairs.
- **Both are needed:** `g-both` is the closest to attention, and the 'gap closed' line says how close a linear-cost layer gets.
- A difference smaller than the seed spread is noise. This is 5 epochs: a pilot, not a final result.
"""),
            *DOWNLOAD]


def hybrid():
    return [md("""
# Hybrid: how much does one attention layer recover?

6-layer models where layer 6 (index 5) is full attention and the other five use a local mixer. The local-only versions are trained alongside, so we can measure **how much of each local mixer's gap to attention one attention layer recovers**. One extra model puts the attention layer in the middle (index 2) instead.

| local mixer | local only | + attention at layer 6 |
|---|---|---|
| `grassmann` (paper) | ✓ | ✓ |
| `concat32` (no geometry, same compression) | ✓ | ✓ (also at layer 3) |
| `hyena` (short filter) | ✓ | ✓ |
| `g-both` (edge-choosing, reach 127) | ✓ | ✓ |
| `attention` (all six layers) | ✓ | |

2 seeds × 5 epochs at LR 1e-3, validation only. About an hour on an A100 (estimate).
"""),
            *setup("hybrid.jsonl"),
            TRAIN_HELPER,
            code("""
CONFIGS = [('attention', ''),
           ('grassmann', ''), ('grassmann', '5'),
           ('concat32', ''), ('concat32', '5'), ('concat32', '2'),
           ('hyena', ''), ('hyena', '5'),
           ('g-both', ''), ('g-both', '5')]
SEEDS = [0, 1]
for seed in SEEDS:
    for m, attn in CONFIGS:
        train(m, seed, attn=attn)
"""),
            md("## Results\n\n'vs attention' is per seed; positive means worse than the all-attention model."),
            code("""
from grassman import analyze
rows = analyze.load(RUNS)
print(analyze.val_table(rows, 'attention'))

mean = lambda n: sum(r['best_val_ppl'] for r in rows if analyze.name(r) == n) / max(1, sum(analyze.name(r) == n for r in rows))
a = mean('attention')
print('\\nshare of the gap to attention recovered by one attention layer:')
for m in ['grassmann', 'concat32', 'hyena', 'g-both']:
    local, hyb = mean(m), mean(m + '+attn@5')
    print(f"  {m:<10} local {local:6.1f} -> hybrid {hyb:6.1f}   recovered {100 * (local - hyb) / (local - a):4.0f}%")
print(f"  concat32 with attention at layer 3 instead of 6: {mean('concat32+attn@2'):.1f}")
"""),
            CURVES,
            md("""
## How to read it

- **Recovered ≥ 80%:** one attention layer does most of the work, and the local mixer is a cheap supporting part, as in StripedHyena and Jamba.
- **Local mixer type inside a hybrid:** if `grassmann`, `concat32`, `hyena` and `g-both` hybrids are all close, use the cheapest one. That is the answer for the Hyena-audio striped design too.
- **Layer 3 vs layer 6:** whether the attention layer's position matters.
- 5 epochs, 2 seeds: a pilot, not a final result.
"""),
            *DOWNLOAD]


def readouts():
    return [md("""
# Readouts: is the edge-choosing layer interpretable?

`g-both-angle` chooses, for each token in each layer, a weighting over 6 edges (1, 2, 4, 16, 64 and 127 tokens back). It can also use sin(angle) between the two tokens: 0 = same direction, 1 = completely different. This notebook trains it (2 seeds × 5 epochs, saving the weights) and runs three tests:

1. **Readable:** which edges each layer uses, and how peaked the choice is. Are the patterns the same across seeds?
2. **Used:** remove one edge at a time and measure how much worse the model gets. Edges that look important should matter when removed.
3. **Meaningful:** is sin(angle) between a token and the previous one higher right after a sentence ends?

About 15 minutes on an A100 (estimate). Validation only.
"""),
            *setup("readouts.jsonl"),
            TRAIN_HELPER,
            code("""
SEEDS = [0, 1]
CKPT = {s: f'{OUTDIR}/ckpt/g-both-angle-s{s}.pt' for s in SEEDS}
for s in SEEDS:
    train('g-both-angle', s, save=CKPT[s], tag='readout')
"""),
            code("""
import torch
from grassman import data, readout
dev = torch.device('cuda')
val_x, val_y = data.blocks(data.load(DATA)['validation'], 128)
models = {s: readout.load(CKPT[s], dev) for s in SEEDS}
for s, m in models.items():
    print(f'seed {s}: validation perplexity of the saved weights {torch.exp(torch.tensor(readout.mean_nll(m, val_x, val_y))):.1f}')
"""),
            md("## 1. Readable: edge use per layer\n\n`share` is the average weight on each offset (positions where all six edges exist). `top` is the average weight of each token's strongest edge: 1/6 ≈ 0.17 would be no preference, 1.0 always one edge."),
            code("""
import numpy as np
usage = {s: readout.edge_usage(m, val_x) for s, m in models.items()}
deltas = usage[SEEDS[0]][0]['deltas']
print('layer  seed  ' + ''.join(f't-{d:<6}' for d in deltas) + '  top')
for layer in usage[SEEDS[0]]:
    for s in SEEDS:
        u = usage[s][layer]
        print(f'{layer:>5}  {s:>4}  ' + ''.join(f'{x:<8.2f}' for x in u['share']) + f"  {u['top_weight']:.2f}")
same = [np.corrcoef(usage[SEEDS[0]][l]['share'], usage[SEEDS[1]][l]['share'])[0, 1] for l in usage[SEEDS[0]]]
print('\\nagreement between seeds per layer (correlation of shares):', ' '.join(f'{c:.2f}' for c in same))
"""),
            code("""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(SEEDS), figsize=(5 * len(SEEDS), 3.5))
for ax, s in zip(np.atleast_1d(axes), SEEDS):
    grid = np.array([usage[s][l]['share'] for l in usage[s]])
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=grid.max(), aspect='auto')
    ax.set_xticks(range(len(deltas)), [f't-{d}' for d in deltas]); ax.set_yticks(range(len(grid)), [f'layer {l}' for l in usage[s]])
    ax.set_title(f'edge share, seed {s}')
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='w' if v < grid.max() / 2 else 'k', fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUTDIR}/readout_edge_share.png', dpi=150); plt.show()
"""),
            md("## 2. Used: remove one edge at a time\n\nIncrease in validation loss (nats per token) on the first 300 validation blocks. Compare with the shares above: an edge with a large share and a large increase is genuinely used."),
            code("""
abl = {s: readout.edge_ablation(m, val_x[:300], val_y[:300]) for s, m in models.items()}
for s in SEEDS:
    print(f\"seed {s}: base perplexity {abl[s]['base_ppl']:.1f}\")
    print('layer  ' + ''.join(f't-{d:<7}' for d in deltas))
    for layer, row in abl[s]['delta_nll'].items():
        print(f'{layer:>5}  ' + ''.join(f'{row[d]:<+9.3f}' for d in deltas))
"""),
            md("## 3. Meaningful: does sin(angle) mark sentence boundaries?\n\nMean sin(angle) between z_t and z_{t−1} when token t−1 is `.`, `!` or `?`, against everywhere else. The learned `angle_weight` shows whether each layer's edge scores use the angle at all."),
            code("""
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
ends = tok.convert_tokens_to_ids(['.', '!', '?'])
for s, m in models.items():
    sb = readout.sin_by_boundary(m, val_x, ends)
    print(f'seed {s}')
    for layer, v in sb.items():
        w = readout.pair_layers(m)[layer][1].angle_weight.item()
        print(f\"  layer {layer}: after a sentence end {v['after_boundary']:.3f}   elsewhere {v['elsewhere']:.3f}   \"
              f\"(n={v['n_boundary']:,}, angle weight {w:+.2f})\")
"""),
            md("""
## How to read it

- **Readable:** a high `top` value (well above 0.17) and the same pattern in both seeds mean each layer has a stable, nameable job ("layer 0 reads the previous token").
- **Used:** the edges with the biggest shares should also cost the most when removed. If removing them changes nothing, the readout is decoration.
- **Meaningful:** sin(angle) clearly higher after sentence ends means the angle is a 'something changed here' signal. If the angle weights stay near 0, the model ignores it.
"""),
            *DOWNLOAD]


def write(name, cells):
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    (HERE / name).write_text(json.dumps(nb, indent=1))
    print("wrote", name, len(cells), "cells")


if __name__ == "__main__":
    write("speed.ipynb", speed())
    write("choose_vs_reach.ipynb", choose_vs_reach())
    write("hybrid.ipynb", hybrid())
    write("readouts.ipynb", readouts())
