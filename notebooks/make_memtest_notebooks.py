"""Write the Colab notebooks for the memory tests (memtest/PROTOCOL.md).

    python notebooks/make_memtest_notebooks.py

memory_test.ipynb   one notebook, two parts:
                    Track A: the associative matrix as a store, vs FAISS (PQ at equal memory, HNSW, exact)
                    Track B: memory layers inside a Hyena model on long-range recall

Results go under /content/results (store.jsonl, recall.jsonl).
"""
import re

from make_notebooks import DOWNLOAD, code, md, setup, write


def _setup(runs_file, extra_pip=""):
    cells = setup(runs_file, data=False)
    if extra_pip:
        cells.insert(2, code(f"!pip install -q {extra_pip}"))
    return cells


def store_test():
    return [md("""
# Track A: is linear attention's associative matrix a useful store?

Store N items (key vector → item id) in one fixed d×d matrix, look them up by a query, and compare with FAISS. Every compressed method gets **the same bytes** as the matrix; product quantisation (PQ) at equal memory is the fair competitor. Protocol, decision rules and predictions: `memtest/PROTOCOL.md` (written before any run).

| stage | what | rough time |
|---|---|---|
| A1 capacity | random keys, d = 64 / 256 / 1024, N from d/8 to 8d, 5 seeds | longest stage (delta-rule loops at d = 1024) |
| A2 real data | Quora duplicate questions, MiniLM-L6 embeddings, 3 budgets, dev then test | minutes |
| A3 updates | overwrite / delete items, stale answers, float drift | minutes |
| A5 cost | write and read time, N up to 300,000 | minutes |

**Runtime:** CPU is enough; a T4 only speeds up the embedding step. Times are not measured yet.

Each stage writes a `.done` marker, so after a disconnect re-run all cells and finished stages are skipped (set `RESTORE = True` and upload `results.zip` if the runtime was deleted).
"""),
            *_setup("store.jsonl", "faiss-cpu sentence-transformers"),
            code("""
EMB = '/content/data/quora_emb.npz'
def stage(name, args):
    marker = f'{OUTDIR}/store_{name}.done'
    if os.path.exists(marker):
        print('skip', name, '(done)'); return
    subprocess.run([sys.executable, '-m', 'memtest.store', *args, '--out', RUNS, '--emb', EMB], cwd=REPO, check=True)
    open(marker, 'w').close()
"""),
            md("Embed the Quora pairs once (dedupe, split into calibration / dev / test, 20,000 pairs each)."),
            code("""
if not os.path.exists(EMB):
    subprocess.run([sys.executable, '-m', 'memtest.store', 'embed', '--emb', EMB], cwd=REPO, check=True)
"""),
            md("## A1: capacity on random keys"),
            code("stage('capacity', ['capacity'])"),
            md("## A2: real data. Dev first (the matrix variant is chosen here), then test (used once)."),
            code("stage('real_dev', ['real', '--split', 'dev'])"),
            code("stage('real_test', ['real', '--split', 'test'])"),
            md("## A3: updates and deletes"),
            code("stage('updates', ['updates'])"),
            md("## A5: cost against N"),
            code("stage('cost', ['cost'])"),
            md("## Results"),
            code("""
out = subprocess.run([sys.executable, '-m', 'memtest.store', 'summary', '--out', RUNS], cwd=REPO,
                     capture_output=True, text=True, check=True).stdout
open(f'{OUTDIR}/store_summary.md', 'w').write(out)
from IPython.display import Markdown, display
display(Markdown(out))
"""),
            code("""
import json, matplotlib.pyplot as plt, numpy as np
from collections import defaultdict
rows = [json.loads(l) for l in open(RUNS) if '"exp": "real"' in l]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, dv in zip(axes, (64, 384, 1536)):
    by = defaultdict(list)
    for r in rows:
        if r['split'] == 'test' and r['dv'] == dv and r['recall1'] is not None:
            by[r['method']].append((r['N'], r['recall1']))
    for m in ('add+c', 'delta+c', 'lstsq+c', 'pq', 'fifo', 'flat'):
        if by[m]:
            Ns = sorted({n for n, _ in by[m]})
            ax.plot(Ns, [100 * np.mean([v for n, v in by[m] if n == N]) for N in Ns], marker='o', label=m)
    ax.set_xscale('log'); ax.set_title(f'test, {384 * dv * 4 / 1e3:.0f} KB'); ax.set_xlabel('items stored (N)')
    ax.grid(alpha=.3)
axes[0].set_ylabel('recall@1 %'); axes[0].legend()
plt.savefig(f'{OUTDIR}/store_real.png', dpi=150, bbox_inches='tight'); plt.show()
"""),
            md("""
## How to read it (decision rules from `memtest/PROTOCOL.md`)

- **Dropped as a store** if the chosen matrix variant is worse than PQ at N = 384 and N = 768 (Holm p < 0.05).
- **"Useful memory tier"** only if it is within 2 points of PQ, or better, at every N ≥ 384.
- `flat` and `hnsw` have no budget: they show the best any method could do, not a fair comparison.
"""),
            *DOWNLOAD]


RECALL_HELPER = code("""
def train(model, task, T, pairs, seed, lr, steps, batch, tag, test=False):
    cmd = [sys.executable, '-m', 'memtest.train', '--model', model, '--task', task, '--T', str(T),
           '--pairs', str(pairs), '--seed', str(seed), '--lr', str(lr), '--steps', str(steps),
           '--batch', str(batch), '--tag', tag, '--out', RUNS, '--skip-existing']
    if not test:
        cmd += ['--no-test']             # pilot and LR sweep never touch the test set
    subprocess.run(cmd, cwd=REPO, check=True)

def analyze(*args):
    print(subprocess.run([sys.executable, '-m', 'memtest.analyze', RUNS, *map(str, args)], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout)
""")


def recall_test():
    return [md("""
# Track B: does a memory layer give Hyena long-range recall?

Small 4-layer models (d = 128) on two synthetic recall tasks. The question is whether a cheap memory layer (a fixed delta-rule matrix, or a lookup over chunk summaries, "moment memory") recovers what one attention layer gives a Hyena model, at a fraction of the memory. Protocol, decision rules and predictions: `memtest/PROTOCOL.md` (written before any run).

| model | layers | what the memory layer is |
|---|---|---|
| `hyena` | HHHH | none (long implicit Hyena filters only) |
| `hyena+attn` | HHAH | one full attention layer |
| `hyena+linear` | HHLH | linear attention: one fixed matrix per head |
| `hyena+delta` | HHDH | delta-rule matrix (DeltaNet) |
| `hyena+moment` | HHMH | lookup over 4 summary slots per 64-token chunk |
| `attention` / `swa` | AAAA / WWWW | references (full / window 128) |
| `hyena+attn@0`, `@3` | AHHH / HHHA | exploratory: where the attention layer goes |

Tasks: **far**, where 8 key-value pairs are scattered in noise and asked for at the end, measuring accuracy by distance (T = 1024 and 4096); and **mqar** (Zoology), with 64 pairs packed at the start (T = 512).

**Use an A100** and run the whole notebook on one GPU type. Times are not measured yet: **check the pilot's time per run before starting the main stage.** Finished runs are skipped on re-run; after a deleted runtime set `RESTORE = True` and upload `results.zip`.
"""),
            *_setup("recall.jsonl"),
            RECALL_HELPER,
            code("""
MAIN = ['hyena', 'hyena+attn', 'hyena+linear', 'hyena+delta', 'hyena+moment', 'attention', 'swa']
PLACEMENT = ['hyena+attn@0', 'hyena+attn@3']
STEPS = 8000
# same tokens per step at every length: 64 x 1024 = 16 x 4096 = 128 x 512
BATCH = {512: 128, 1024: 64, 4096: 16}
"""),
            md("""
## Stage 0: pilot (dev only)

Two models, 2,000 steps. **Check before going on:** `attention` should reach > 90% dev accuracy; if it does not, the step count is too low for this task (raise `STEPS` and note it in the log). The printed time per run tells you how long the rest will take (there are 21 sweep runs and 63 main runs).
"""),
            code("""
import time
for m in ['attention', 'hyena']:
    t0 = time.time()
    train(m, 'far', 1024, 8, 0, 1e-3, 2000, BATCH[1024], 'pilot')
    print(f'{m}: {time.time() - t0:.0f} s for 2,000 steps')
analyze('--tag', 'pilot', '--dev')
"""),
            md("## Stage 1: learning-rate sweep (far, T = 1024, seed 0, dev only)"),
            code("""
LRS = [3e-4, 1e-3, 3e-3]
for m in MAIN + PLACEMENT:
    for lr in LRS:
        train(m, 'far', 1024, 8, 0, lr, STEPS, BATCH[1024], 'sweep')
best = json.loads(subprocess.check_output([sys.executable, '-m', 'memtest.analyze', RUNS, '--tag', 'sweep',
                                           '--best-lr'], cwd=REPO, text=True))
print('best learning rate per model:', best)
analyze('--tag', 'sweep', '--dev')
"""),
            md("""
## Stage 2: main runs (3 seeds, test set once)

Seed-major, so an interruption still leaves complete seeds. The number of seeds is fixed by the protocol: do not add seeds after looking at results.
"""),
            code("""
SEEDS = [0, 1, 2]
for seed in SEEDS:
    for m in MAIN:
        train(m, 'far', 1024, 8, seed, best[m], STEPS, BATCH[1024], 'main', test=True)
        train(m, 'far', 4096, 8, seed, best[m], STEPS, BATCH[4096], 'main', test=True)
    for m in MAIN + PLACEMENT:
        train(m, 'mqar', 512, 64, seed, best[m], STEPS, BATCH[512], 'main', test=True)
"""),
            md("## Results\n\n**Primary (confirmatory):** far recall at T = 4096, queries ≥ 1024 tokens back."),
            code("analyze('--tag', 'main', '--task', 'far', '--T', 4096, '--confirmatory', '--min-dist', 1024)"),
            md("Secondary (exploratory): far at T = 1024, and MQAR (all distances)."),
            code("analyze('--tag', 'main', '--task', 'far', '--T', 1024, '--min-dist', 256)"),
            code("analyze('--tag', 'main', '--task', 'mqar', '--T', 512, '--min-dist', 0)"),
            code("""
import matplotlib.pyplot as plt, numpy as np
from memtest import analyze as A, tasks
rows = A.load(RUNS, 'main', 'far', 4096)
fig, ax = plt.subplots(figsize=(8, 4.5))
mids = [np.sqrt(max(1, a) * b) for a, b in zip(tasks.BUCKETS[:-1], tasks.BUCKETS[1:])]
for name, rs in A._group(rows).items():
    c = np.sum([r['test_detail']['bucket_correct'] for r in rs], 0)
    t = np.sum([r['test_detail']['bucket_total'] for r in rs], 0)
    keep = t > 0
    ax.plot(np.array(mids)[keep], 100 * c[keep] / t[keep], marker='o', label=f'{name} ({len(rs)} seeds)')
ax.set_xscale('log'); ax.set_xlabel('distance from key to query (tokens)'); ax.set_ylabel('test accuracy %')
ax.set_title('far recall, T = 4096'); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.savefig(f'{OUTDIR}/recall_by_distance.png', dpi=150, bbox_inches='tight'); plt.show()
"""),
            md("""
## How to read it (decision rules from `memtest/PROTOCOL.md`)

- **"Moment memory adds recall":** hyena+moment − hyena positive, Holm p < 0.05, 3/3 seeds.
- **"Matches attention":** the 95% interval of hyena+moment − hyena+attn lies within ±5 points.
- The `state (MB, bf16)` column is the memory each model keeps per sequence to generate the next token; that is the trade-off this test is about.
"""),
            *DOWNLOAD]


def _rename(cells, new):
    """Point a part's cells at its own results file (RUNS -> new)."""
    for c in cells:
        c["source"] = [re.sub(r"\bRUNS\b", new, line) for line in c["source"]]
    return cells


def memory_test():
    store, recall = store_test(), recall_test()
    n_store_setup, n_recall_setup = 7, 6        # setup() cells, with and without the extra pip cell
    return [md("""
# Memory tests: the associative matrix as a store (Track A), and memory layers inside Hyena (Track B)

Protocol, decision rules and predictions, written before any run: `memtest/PROTOCOL.md`.

- **Track A** (CPU work, minutes to about an hour): is linear attention's fixed matrix a useful store? It is compared with FAISS at **equal memory** (PQ), plus HNSW and exact search as ceilings. Results: `/content/results/store.jsonl`.
- **Track B** (GPU work, the long part): does a cheap memory layer (delta rule, or a lookup over chunk summaries) give a Hyena model long-range recall like one attention layer, with less memory? Results: `/content/results/recall.jsonl`.

**Use an A100 runtime** for the whole notebook (Track A runs on its CPU). Times are not measured yet: check Track B's pilot cell before its main stage. Finished stages and runs are skipped on re-run; after a deleted runtime set `RESTORE = True` and upload `results.zip`.
"""),
            *setup("store.jsonl", data=False)[:2],
            code("!pip install -q faiss-cpu sentence-transformers"),
            *setup("store.jsonl", data=False)[2:],
            code("STORE_RUNS, RECALL_RUNS = f'{OUTDIR}/store.jsonl', f'{OUTDIR}/recall.jsonl'"),
            md("# Track A: the matrix as a store"),
            *_rename(store[1 + n_store_setup:-2], "STORE_RUNS"),
            md("# Track B: memory layers inside Hyena"),
            *_rename(recall[1 + n_recall_setup:-2], "RECALL_RUNS"),
            *DOWNLOAD]


if __name__ == "__main__":
    write("memory_test.ipynb", memory_test())
