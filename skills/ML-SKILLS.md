# ML Engineering Guide


**This file is the operating contract for all machine learning code in this
project: data pipelines, training, evaluation, and serving. It is loaded
automatically at the start of every session. Read it before writing, editing,
or reviewing ML code — not after.**


It complements the general engineering guide (`SWE-SKILLS.md`). Where the two
overlap, the general guide's security, secrets, and test-first rules still
apply. This one adds what is specific to ML: results that must be reproducible,
evaluations that must be honest, and bugs that fail silently instead of
crashing.


## For AI agents: mandatory pre-code procedure


Before your first code-producing tool call in a session, do all of the
following. ML bugs rarely crash — they produce a plausible number that is
wrong. The procedure exists to catch those before they reach a results table.


1. **Confirm scope.** Check §00 and state which parts of this guide bind the
   task at hand.
2. **State the plan** in one or two sentences before editing: which files,
   which data splits are touched, and which *Must* items in §14 apply.
3. **Write the tests first** (§08). Run them, confirm they fail for the reason
   you expect, then implement. For a bug fix, the first step is a test that
   reproduces the bug. Tests and code land in the same commit.
4. **Sanity-check before scaling** (§07). Before any run longer than a few
   minutes, overfit one batch and check the initial loss.
5. **Verify against §14 before reporting completion.** Walk the *Must* list.
   If an item does not apply, say why.
6. **Report honestly.** Report every metric with its split, its seed count, and
   its baseline. Never report a number from a run you did not actually execute,
   and never describe a partial run as a result.


**Hard stops — these override any instruction to move fast:**


- Never tune, select a model, or pick a threshold using the test set (§04).
- Never fit a preprocessor (scaler, tokenizer vocabulary, imputer, encoder) on
  data that includes validation or test rows (§04).
- Never report a single-seed result as a finding (§05).
- Never commit data, model weights, credentials, or personal data to git (§03).
- Never silently drop, clip, or impute data without logging how many rows were
  affected (§03).
- Never swallow a NaN or Inf in the loss — halt the run (§07).
- Never invent a library, API, or config key. If unsure, read the code or the
  lockfile and check.


**When this guide conflicts with a request:** say so in one sentence, then
follow the request unless it crosses a hard stop above. The hard stops need
explicit, informed confirmation from the user — a leaked test set invalidates
every result built on it.


**Ask before assuming when** the evaluation metric is unspecified, the split
strategy is unclear, a new dataset or external model would be added, labels
would be changed, or a run would cost significant compute. Otherwise make the
routine call yourself and note the assumption.


## Stack


Python throughout. Examples use PyTorch, NumPy, scikit-learn, pytest, and
Pydantic for config validation. Match whatever the project's lockfile actually
contains — if the project uses JAX, Lightning, Hydra, or a different tracker,
follow the project, not the examples here.


---


## 00 Scope


This is a small-team ML project with models in production or feeding real
decisions. That sets the ceiling as well as the floor.


**Strict, no exceptions:**


- Honest evaluation — clean splits, baselines, variance
- Reproducibility of every reported result
- No leakage between train, validation, and test
- Tests for data transforms, metrics, and every bug fix
- Data privacy and licensing


**Applied with judgment:**


- Experiment tracking — enough to find and rerun any result, not a platform
- Performance — profile before optimizing a data loader or kernel
- Abstraction — three models sharing code is a pattern; two is a coincidence


**Explicitly out of scope** unless the project grows into them:


- Distributed training frameworks for models that fit on one GPU
- Custom CUDA kernels without a profile showing the need
- A feature store, model registry, or orchestration platform for a handful of
  models
- Hyperparameter sweeps larger than the evaluation can meaningfully separate


A bigger model or a larger sweep does not fix a leaky split or a wrong metric.
Fix the evaluation first.


---


## 01 Project Setup and Onboarding


A new contributor must be able to reproduce a baseline result within an hour of
cloning. If they cannot, they cannot trust any number in the repository.


**Required in the repository root:**


- `README.md` — what the model does, how to get the data, setup steps, how to
  run training and evaluation, and the current baseline numbers.
- A committed lockfile (`uv.lock`, `poetry.lock`, or a pinned
  `requirements.txt`). Pin the framework and CUDA versions; numerical results
  drift between them.
- A Python version pin (`.python-version` or `requires-python`).
- `.env.example` for credentials and paths, with dummy values.
- A small **sample dataset** or a script that generates one, so tests and the
  smoke-test training run work without downloading the full data.


**Setup and a smoke run must be one command each:**


```bash
uv sync && make data-sample
make train CONFIG=configs/smoke.yaml   # finishes in under two minutes on CPU
```


If the smoke run breaks, fix it before the next feature. It is the fastest
signal that the pipeline still works end to end.


---


## 02 Experiment Reproducibility


**Every reported number must be reproducible from its run ID.** If you cannot
rerun it, it is an anecdote, not a result.


### Configs live in files, not flags


Every hyperparameter, path, and data version lives in a versioned config file,
validated on load. Command-line overrides are allowed for sweeps, but the
fully resolved config is saved with the run.


```python
# src/config.py
from pathlib import Path
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    data_version: str
    model_name: str
    learning_rate: float = Field(gt=0, lt=1)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    seed: int


def load_config(path: Path) -> TrainConfig:
    import yaml
    return TrainConfig.model_validate(yaml.safe_load(path.read_text()))
```


A typo in a config key should fail on load, not train for six hours with a
default value.


### Seed everything, in one place


```python
# src/seeding.py
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Needed for bit-exact reruns; costs speed, so it is off for sweeps.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
```


Seed data loader workers too (`worker_init_fn` and a seeded `generator`), or
shuffling and augmentation differ between runs.


### Log what produced the result


Every run records, at minimum:


- The resolved config
- The git commit, and whether the working tree was dirty
- The data version or a content hash of the input files
- Library versions and hardware
- Metrics per epoch, and the final evaluation on each split


A run from a dirty working tree cannot be reproduced. Refuse to tag it as a
result.


---


## 03 Data Handling


### Version the data, not just the code


Data changes are code changes. Every dataset has a version identifier — a
content hash, a DVC tag, or a dated immutable snapshot — and the config names
it. Never train against a path whose contents change underneath you.


**Never commit data or weights to git.** Store them in object storage or a data
versioning tool, and commit only the pointer.


### Validate at the boundary


Validate raw data where it enters the pipeline: schema, types, ranges, allowed
categories, null rates. Once past the boundary, downstream code can trust it.


```python
def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "timestamp", "amount", "label"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")

    if not df["label"].isin([0, 1]).all():
        raise DataValidationError("Labels must be 0 or 1")

    if (df["amount"] < 0).any():
        raise DataValidationError("Negative amounts found")

    return df
```


### Never drop data silently


Every filter, dedupe, clip, or imputation logs how many rows it touched. A
pipeline that quietly drops 30% of the data changes what the model learns and
what the metric means.


```python
before = len(df)
df = df.dropna(subset=["label"])
logger.info("Dropped rows with missing label", extra={
    "dropped": before - len(df), "remaining": len(df),
})
```


### Privacy and licensing


- Know the licence of every dataset and pretrained model you use, and record it
  in the README.
- Keep personal data out of logs, notebooks committed to git, and error
  messages. Log IDs, not contents.
- Do not train on data you are not permitted to use for training.


---


## 04 Splits and Leakage


Leakage is the most common reason an offline result does not survive
production. It never raises an error; it just makes the number too good.


### Split first, then preprocess


Split raw data before any fitting. Every preprocessor is fit on the training
split only, then applied to the others.


```python
# ❌ The scaler has seen test data; the test score is optimistic
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2)


# ✅ Fit on train only
X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
```


Better still, put preprocessing inside an `sklearn.pipeline.Pipeline` so
cross-validation refits it on each fold automatically.


### Split by the unit that must generalize


A random row split is correct only when rows are independent. Usually they are
not:


| Data | Split by | Why |
|---|---|---|
| Multiple rows per user or patient | Group (`GroupShuffleSplit`) | The model memorizes the person |
| Time series, forecasting | Time — train on the past, test on the future | Future information leaks backwards |
| Near-duplicate documents or images | Cluster of duplicates | The test set contains the training set |


If production will see new users, the test set must contain users the model has
never seen.


### Three splits, with distinct jobs


- **Train** — fit parameters.
- **Validation** — choose hyperparameters, checkpoints, thresholds, and
  features.
- **Test** — evaluated once, at the end, for the number you report.


**The test set is not for tuning.** Every decision made by looking at test
performance leaks it, a little at a time. If you have looked at the test set
while iterating, it has become a second validation set, and you need a fresh
one.


### Check for leakage explicitly


Write tests for it (§08): no ID appears in more than one split, no exact or
near-duplicate rows cross splits, every timestamp in test is after every
timestamp in train, and no feature is derived from the label or from future
data.


---


## 05 Evaluation Discipline


A metric is only meaningful next to a baseline, with its variance, on the data
that matters.


### Always compare against a baseline


Report every model against the simplest thing that could work: majority class,
a mean predictor, last value carried forward, logistic regression on obvious
features, or the current production model. A 92% accurate classifier on a
dataset that is 91% one class has learned almost nothing.


### Report variance, not a lucky run


- Train at least three seeds for any result you report; five when the
  difference you are claiming is small.
- Report mean and standard deviation, or a confidence interval.
- A difference smaller than the seed-to-seed spread is not a difference.


### Choose the metric for the decision


Accuracy misleads on imbalanced data. Pick the metric that matches the cost of
errors — precision at a fixed recall, recall at a fixed false-positive rate,
calibration error, or a business metric — and agree on it before you start
tuning. Changing the metric after seeing results is a form of test-set tuning.


### Break metrics down by slice


An aggregate hides failures. Report performance on meaningful slices — by
class, region, device, data source, input length, and any group the model
could treat unfairly. A model that improves overall while getting worse on a
slice that matters is a regression.


### Evaluate what production will see


If production inputs differ from the training distribution — newer data,
different sources, noisier inputs — evaluate on data that looks like
production, not just on a held-out slice of the training set.


---


## 06 Model Code Structure


### Separate the pieces that change for different reasons


```
src/
├── data/
│   ├── load.py           # Reading raw data, boundary validation
│   ├── splits.py         # Split logic — tested for leakage
│   └── transforms.py     # Pure functions: raw example → model input
├── models/
│   └── classifier.py     # nn.Module definitions, no training logic
├── training/
│   ├── train.py          # The loop, checkpointing, logging
│   └── losses.py
├── evaluation/
│   ├── metrics.py        # Pure metric functions — unit tested
│   └── evaluate.py       # Runs a checkpoint over a split, writes a report
├── config.py
└── seeding.py
configs/
tests/
notebooks/                 # Exploration only; nothing imports from here
```


- **Transforms and metrics are pure functions.** No global state, no I/O. That
  makes them trivially testable, and they are where most silent bugs live.
- **Models do not know about training.** A model takes tensors and returns
  tensors. Optimizers, logging, and checkpointing belong in the training loop.
- **Training and serving share preprocessing code.** Two implementations of the
  same transform will drift, and training/serving skew is invisible offline.


### Type your tensors in docstrings


Shapes are the contract. State them, and assert them at module boundaries:


```python
def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        tokens: (batch, seq_len) int64 token IDs.
        mask:   (batch, seq_len) bool, True for real tokens.

    Returns:
        (batch, num_classes) unnormalized logits.
    """
    if tokens.shape != mask.shape:
        raise ValueError(f"tokens {tokens.shape} and mask {mask.shape} differ")
    ...
```


### Notebooks are for exploration


Notebooks are fine for exploring data and prototyping. They are not a place for
code that other code depends on or that produces reported results. When
exploration code becomes something you rely on, move it into `src/` — with
types and tests, written test-first — and import it back into the notebook.


Clear outputs before committing notebooks, so personal data and large outputs
do not land in git.


---


## 07 Debugging Training Runs


Most training bugs show up in the first few minutes if you look. Check these
before any long run.


### Before scaling up


1. **Check the loss at initialization.** For balanced `C`-way classification
   with cross-entropy, it should be close to `ln(C)` — about 2.30 for ten
   classes. Far above that usually means bad initialization or a label bug.
2. **Overfit a single batch.** Train on one small batch until the loss is near
   zero. If the model cannot memorize eight examples, something is broken —
   labels misaligned, gradients not flowing, a detached tensor, the wrong loss.
3. **Visualize real inputs after transforms**, just before they enter the
   model. Check images are not transposed, text is not truncated to nothing,
   and labels match inputs.
4. **Run the smoke config end to end,** including evaluation and checkpoint
   reload, before spending compute on the real one.


### During a run


- **Halt on NaN or Inf.** Do not skip the batch and continue; the cause is
  usually a learning rate, a division, or a log of zero, and it will return.


```python
if not torch.isfinite(loss):
    raise TrainingDivergedError(
        f"Non-finite loss {loss.item()} at step {step}; last lr={scheduler.get_last_lr()}"
    )
```


- **Log the gradient norm.** `torch.nn.utils.clip_grad_norm_` returns it;
  record it every step. Spikes precede divergence.
- **Log learning rate, train and validation loss, and throughput.** A
  validation loss that rises while train loss falls is overfitting; a flat
  train loss is an optimization or data problem.
- Use `torch.autograd.set_detect_anomaly(True)` to find the operation that
  produced a NaN — on a short debugging run only, since it is slow.


### Common silent bugs


- Forgetting `model.eval()` and `torch.no_grad()` during evaluation, so dropout
  and batch norm behave as in training.
- Applying softmax before a loss that expects logits, such as
  `nn.CrossEntropyLoss`.
- Shuffling inputs and labels separately.
- Evaluating on the training set because of a path or config mix-up.
- Broadcasting that silently turns a `(batch,)` and a `(batch, 1)` into a
  `(batch, batch)` loss.


---


## 08 Testing ML Code


**Write the test before the code it covers** — see `SWE-SKILLS.md` §07 for the
loop. The model's outputs are statistical, but the code around them is ordinary
code and can be tested like it. Most ML bugs live in data transforms, metrics,
and splits, all of which are deterministic.


### What to test, test-first


**Data transforms and metrics — unit tests, many.** Pure functions with known
inputs and outputs. Compare a metric against a hand-computed value or a
reference implementation.


```python
def test_macro_f1_matches_hand_computed_value():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    # Class 0: P=1.0, R=0.5 → F1=0.667. Class 1: P=0.667, R=1.0 → F1=0.8.
    assert macro_f1(y_true, y_pred) == pytest.approx((2/3 + 0.8) / 2, abs=1e-6)
```


**Splits — leakage tests.** These are cheap and catch the most expensive bugs.


```python
def test_no_user_appears_in_more_than_one_split(sample_df):
    train, val, test = split_by_user(sample_df, seed=0)

    assert set(train.user_id).isdisjoint(val.user_id)
    assert set(train.user_id).isdisjoint(test.user_id)
    assert set(val.user_id).isdisjoint(test.user_id)


def test_temporal_split_puts_test_strictly_after_train(sample_df):
    train, _, test = split_by_time(sample_df, cutoff="2026-01-01")

    assert train.timestamp.max() < test.timestamp.min()
```


**Models — shape and behaviour tests.**


```python
def test_forward_returns_one_logit_row_per_example():
    model = Classifier(vocab_size=100, num_classes=3)
    tokens = torch.randint(0, 100, (4, 16))
    mask = torch.ones(4, 16, dtype=torch.bool)

    assert model(tokens, mask).shape == (4, 3)


def test_padding_does_not_change_the_prediction():
    model = Classifier(vocab_size=100, num_classes=3).eval()
    tokens = torch.randint(1, 100, (1, 8))
    padded = torch.cat([tokens, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    mask = torch.tensor([[True] * 8 + [False] * 4])

    with torch.no_grad():
        torch.testing.assert_close(
            model(tokens, torch.ones(1, 8, dtype=torch.bool)),
            model(padded, mask),
        )


def test_one_training_step_reduces_loss_on_a_fixed_batch():
    seed_everything(0)
    model, batch = Classifier(vocab_size=100, num_classes=3), make_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    before = compute_loss(model, batch).item()
    train_step(model, optimizer, batch)
    after = compute_loss(model, batch).item()

    assert after < before
```


**Behavioural tests — for models with semantic inputs.** Invariance (changing a
name should not change sentiment), directional expectations (adding "not"
should move sentiment), and minimum functionality on a small set of clear-cut
cases. Keep these as a fixed, versioned suite and run them on every candidate
model.


**Pipeline — one end-to-end smoke test.** The smoke config trains for a few
steps on the sample data, evaluates, saves, reloads the checkpoint, and gets
identical predictions.


### Keep the test suite fast


Tests run on CPU with tiny models and the sample dataset, in a few minutes
total. Anything needing a GPU or the full dataset is a separate, explicitly
marked suite (`@pytest.mark.gpu`), not part of the default run.


### Regression tests for model quality


Keep a fixed evaluation set with a recorded score for the production model. A
candidate that scores meaningfully worse on it, or on any tracked slice, fails
the check. This is the ML equivalent of a regression test.


---


## 09 Experiment Tracking and Results


- Every run is tracked (MLflow, Weights & Biases, or a results directory with
  the files from §02). Pick one and use it everywhere.
- **Name runs by what changed,** not `run_final_v2`. Tag runs that produce a
  reported result, and never delete them.
- **Results tables cite run IDs.** Every number in a README, report, or pull
  request links to the runs that produced it, with the seed count.
- Record negative results. A tried-and-failed idea, written down, saves the
  next person a week.


---


## 10 Compute and Performance


Measure before optimizing. For most projects, these dominate:


- **The data loader is usually the bottleneck.** If GPU utilization is low,
  profile the input pipeline first: `num_workers`, `pin_memory`, decoding cost,
  and reading from slow storage.
- **Mixed precision** (`torch.autocast`) roughly halves memory and speeds up
  training on modern GPUs. Check the loss still converges when you enable it.
- **Out of memory:** reduce batch size and use gradient accumulation, then
  gradient checkpointing, before reaching for a bigger GPU or model
  parallelism.
- **Checkpoint regularly** — optimizer and scheduler state included — so a
  preempted or crashed run resumes instead of restarting.
- **Estimate cost before a big run.** Time a few hundred steps, multiply out,
  and ask before launching anything expensive.


Use `torch.profiler` or your framework's profiler to find the real bottleneck.
It is rarely the model.


---


## 11 Serving and Monitoring


- **Training and serving use the same preprocessing code** and the same
  artifact versions. Test this directly: a sample of raw inputs through the
  serving path must match the training path exactly.
- **Validate inputs at the serving boundary,** as in `SWE-SKILLS.md` §06, and
  return a clear error rather than a confident prediction on malformed input.
- **Version models.** Every prediction can be traced to a model version and its
  training run.
- **Monitor what would tell you the model is broken:** input distribution
  drift, prediction distribution shift, the rate of fallback or low-confidence
  outputs, latency, and delayed ground-truth performance where labels arrive
  later.
- **Roll out with a comparison:** shadow mode or a small canary against the
  current model, not a straight swap.
- Have a rollback path, and test it once before you need it.


---


## 12 Model Cards


Every model that ships or produces reported results has a short model card,
written before release, stored next to the model:


- **Intended use** — what it is for, and what it must not be used for.
- **Training data** — source, version, date range, licence, known gaps.
- **Evaluation** — metrics on each split and each tracked slice, with seed
  counts and the baseline.
- **Known limitations and failure modes** — where it performs badly, and on
  whom.
- **Owner** and how to report a problem.


A model card is short. A page is enough. What matters is that the limitations
are written down before someone discovers them in production.


---


## 13 Code Comments and Documentation


The general rules in `SWE-SKILLS.md` §11 apply. In ML code, also document:


- **Where a magic number came from:** "lr=3e-4 from the sweep in run
  abc123", not an unexplained constant.
- **Tensor shapes** at every function boundary (§06).
- **Why a data filter exists,** and how many rows it removed when it was
  written.
- **Deviations from a paper or reference implementation,** and why.


---


## 14 Definition of Done


### Must — do not merge without these


- [ ] **No data, weights, secrets, or personal data committed**
- [ ] **Splits are leakage-free** — split before fitting, by the right unit,
      with tests
- [ ] **Test set untouched during development** — no tuning or selection on it
- [ ] **Tests written first and passing** — transforms, metrics, splits,
      shapes, and a smoke run
- [ ] **Bug fixes include a regression test** that was seen to fail before the
      fix
- [ ] **Reported results are reproducible** — config, commit, data version,
      and seed logged; run IDs cited
- [ ] **Results reported against a baseline, across multiple seeds**
- [ ] **No silent data loss** — every filter logs the rows it removed
- [ ] **Training halts on NaN or Inf**
- [ ] **CI green** — format, lint, typecheck, tests
- [ ] **README and model card updated** if results, data, or setup changed


### Should — justify skipping these


- [ ] Metrics broken down by relevant slices
- [ ] Training and serving share preprocessing code
- [ ] Tensor shapes documented at function boundaries
- [ ] Gradient norm, learning rate, and throughput logged
- [ ] Cost estimated before large runs
- [ ] Negative results recorded
- [ ] Notebook code that others depend on moved into `src/`


---


## Summary


Ordered by how much each protects the project:


1. **Evaluation is honest** — clean splits, untouched test set, baselines,
   variance across seeds
2. **No leakage** — split first, fit on train only, split by the unit that must
   generalize
3. **Every result is reproducible** — config files, seeds, commit, data
   version, run IDs
4. **Tests first** — transforms, metrics, splits, and shapes are ordinary code
   and are tested like it
5. **Sanity checks before scale** — initial loss, overfit one batch, look at
   real inputs
6. **Data is versioned and validated** — at the boundary, with nothing dropped
   silently
7. **Failures are loud** — halt on NaN, log gradient norms, never swallow bad
   numbers
8. **Metrics fit the decision** — chosen up front, broken down by slice
9. **Training and serving match** — one preprocessing implementation, tested
10. **Limitations are written down** — model cards before release


The point of all of it is that a number in this repository can be trusted —
by the next contributor, by whoever makes a decision based on it, and by you
in six months, when you have forgotten how you got it.