# Research note: evaluating RAG answer quality offline

A lab-style note on how this repository measures answer quality without a live
LLM, why the naive string metrics are not enough on their own, and what the
ensemble judge, calibration, composite score and pairwise-AUC separation add.
Every number below is reproduced by a command in this document; the commands
run offline (hash embedder, echo provider, no network).

## Observation

The three per-answer string metrics in `app/evals/metrics.py` -
`keyword_recall`, `numbers_preserved`, `grounding` - are cheap and deterministic,
but two of them saturate on the golden set. Running the plain report:

```bash
python -m app.evals
```

```
cases       : 17
pass_rate   : 1.000 (threshold 0.500)
per-metric averages:
  grounding         : 0.732
  keyword_recall    : 1.000
  numbers_preserved : 1.000
```

`keyword_recall` and `numbers_preserved` both average `1.000`. On this corpus
they are pinned at the ceiling: an answer that echoes the expected keywords and
copies the numbers scores a perfect `1.000` on both, whether or not it is
actually supported. A metric stuck at `1.000` carries no separating signal - it
cannot tell a grounded answer from a leaked one. `grounding` is the only string
metric that moves (`0.732`), and even it is symmetric to context order, so a
"self-consistency across shuffles" check on it is trivially satisfied and adds
nothing.

## Hypothesis

An ensemble of independent weak judges, each reading a *different* signal, plus a
weighted composite score, a calibration measure (ECE) and a pairwise-AUC
separation, gives a more robust and more honest quality signal than any single
string metric - specifically, it stays informative on inputs where an individual
metric has saturated.

## Method

All offline and deterministic. Modules (verified present in this tree):

- `app/evals/judge.py` - `EnsembleJudge`. Three weak judges over
  `app/evals/metrics.py`: a grounding judge, a numeric-preservation judge and a
  keyword judge. Each votes good/bad by a threshold; `score` is the weighted mean
  of the good-votes and `confidence` is the fraction of judges agreeing with the
  majority (all agree -> `1.000`; split -> lower). The disagreement is real:
  different judges catch different failures.
- `app/evals/calibration.py` - `expected_calibration_error` (binned, weighted
  `|accuracy - confidence|`, in `[0, 1]`, empty input guarded to `0.0`) and
  `reliability_table`. Pure numpy.
- `app/evals/composite.py` - `composite_faithfulness` (a weighted combination of
  grounding, keyword recall and numeric preservation) and `separation`, a strict
  pairwise ROC-AUC: the fraction of (good, bad) pairs correctly ordered, ties
  counted as `0.5` (the Mann-Whitney U statistic).
- `app/evals/retrieval.py` - `hit_at_k` and `mrr`, run over a mini-corpus built
  from the golden `relevant_doc_ids` with `HashEmbedder` + `InMemoryVectorStore`.

## Experiment

Two labelled sets, both offline:

- **Golden set** - `app/evals/golden_set.json`, 17 cases, used for the plain
  per-metric report and for retrieval (its `relevant_doc_ids`).
- **Judged set** - `app/evals/judged_set.json`, 14 hand-written cases
  (6 `good`, 8 `bad`). The answers are written by hand, not produced by the echo
  provider, so the judge sees realistic text. The `bad` cases split along two
  distinct axes:
  - *number corruption* - the answer is grounded in words but states a wrong
    number, so `numbers_preserved == 0.0` while `grounding > 0`. Caught by the
    numeric judge only.
  - *hallucination by words* - the answer reads fluently from a foreign
    vocabulary, so `grounding` is low; its `reference_numbers` is empty, which
    makes `numbers_preserved` vacuously `1.000`, so this case is caught by
    `grounding`, not by the numeric metric.

Reproduction commands:

```bash
python -m app.evals              # plain per-metric report (golden set)
python -m app.evals --judge      # ensemble judge: ECE + per-case confidence + AUC
python -m app.evals --retrieval  # hit@k / MRR over the golden corpus
python -m app.bench              # latency / throughput + batch-call benchmark
```

## Results

All numbers below come from the commands above on this tree (offline).

### Baseline string metrics (leak-prone), `python -m app.evals`

| metric              | golden avg | note                                  |
| ------------------- | ---------: | ------------------------------------- |
| `grounding`         |      0.732 | the only string metric that moves     |
| `keyword_recall`    |      1.000 | saturated - no separating signal      |
| `numbers_preserved` |      1.000 | saturated - no separating signal      |

### Ensemble judge, `python -m app.evals --judge`

14 judged cases; ECE and the separation of each score across the `good`/`bad`
split:

```
cases            : 14
ECE (confidence) : 0.095
AUC separation good>bad (higher = cleaner signal):
  composite : 1.000
  grounding : 0.844
  keyword   : 0.750
  numbers   : 0.750
```

- **ECE = 0.095** - confidence is reasonably calibrated on this set.
- **AUC separation** - the composite reaches `1.000`, above every single metric
  (best single is `grounding` at `0.844`; `keyword` and `numbers` sit at
  `0.750`). The composite orders every good-above-bad pair correctly; the
  individual metrics do not.

The saturated metrics collapse toward chance when a signal has no spread. On a
uniform (leaked) input where every score ties, `separation` returns exactly
`0.5` - reproduced directly:

```bash
python -c "from app.evals.composite import separation; \
print('clean', separation([0.9,0.8,0.7],[0.3,0.2,0.1])); \
print('leak (ties)', separation([1.0,1.0,1.0],[1.0,1.0,1.0]))"
# clean 1.0
# leak (ties) 0.5
```

ECE behaves as expected across a perfect and a badly-calibrated set:

```bash
python -c "from app.evals.calibration import expected_calibration_error as ece; \
print('perfect', ece([1.0,1.0,0.0,0.0],[True,True,False,False])); \
print('bad', ece([0.9,0.9,0.9,0.9],[False,False,False,True])); \
print('empty', ece([],[]))"
# perfect 0.0
# bad 0.65
# empty 0.0
```

### Retrieval quality, `python -m app.evals --retrieval`

```
queries     : 17
hit@3      : 1.000
MRR         : 1.000
```

`hit@3 = 1.000` and `MRR = 1.000` on the golden corpus - with a hash embedder and
in-memory store the relevant chunk is retrieved in the top position for every
query. This is a floor for the retrieval wiring, not a hard task; a real corpus
with near-duplicate documents would push both below `1.000`.

### Batch-call structure, `python -m app.bench`

Batching the embedder does not change the vectors (the output is byte-for-byte
identical to a single `embed(texts)` call); it changes how many provider calls
are made. For 100 texts at `batch_size = 32` the call count drops from **100 to
4** (`ceil(100 / 32) = 4`) - a deterministic structural fact. Reproduced with a
delay-mock that sleeps per `embed` call:

```bash
python -c "
import asyncio, time
from app.rag.embed import embed_in_batches
class DelayEmbedder:
    dim = 8
    def __init__(self): self.calls = 0
    async def embed(self, texts):
        self.calls += 1; await asyncio.sleep(0.001)
        return [[0.0]*self.dim for _ in texts]
async def main():
    texts = ['t%d' % i for i in range(100)]
    e1 = DelayEmbedder(); t0 = time.perf_counter()
    await embed_in_batches(e1, texts, batch_size=32)
    b = (time.perf_counter()-t0)*1000
    e2 = DelayEmbedder(); t0 = time.perf_counter()
    for t in texts: await e2.embed([t])
    o = (time.perf_counter()-t0)*1000
    print('batched  calls=%d wall_ms=%.1f' % (e1.calls, b))
    print('per-item calls=%d wall_ms=%.1f' % (e2.calls, o))
asyncio.run(main())
"
# batched  calls=4   wall_ms=~4.6
# per-item calls=100 wall_ms=~113
```

The call count (**4 vs 100**) is exact and repeats every run. The wall-clock is a
measured illustration, not a fixed number: with a 1 ms simulated latency the
batched path finishes roughly 25x faster because it pays the per-call latency 4
times instead of 100. On the offline hash embedder there is no such gain - it is
the same Python loop either way - so this speed-up is real only for a networked
provider with per-request overhead; here it is measured against a *simulated*
network delay.

## Limitations

- The echo provider is not a real LLM. It concatenates context and question; the
  plain report measures the metrics' behaviour, not a model's answer quality.
- The golden set is small (17 cases) and the judged set smaller (14). Numbers
  like `hit@3 = 1.000` and the AUC figures are indicative on a toy corpus, not
  benchmark results.
- The ensemble judge is a set of string-metric proxies with a voting rule, not a
  learned scorer. Its `confidence` is judge-agreement, not a probability from a
  trained model.
- The batch speed-up is measured against a simulated per-call delay
  (`asyncio.sleep`). The vector output is identical with or without batching; the
  only real-world win is fewer network round-trips.
- The linear probe (`app/ml/`, when present) is a small demo on a synthetic,
  linearly-separable toy dataset - it shows the training loop works, not that the
  embeddings carry rich structure.
