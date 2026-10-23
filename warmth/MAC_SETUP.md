# Native macOS / Apple Silicon setup

Run on the Mac itself, not inside the Linux container. A container running on an Apple Silicon host does not give this PyTorch process the native macOS MPS backend. This pipeline accepts `--device mps`; activation diagnostics use CPU float64. Model inference uses BF16. MPS execution has not been tested in our CPU-only development environment.

Use native arm64 Python 3.12 or 3.13 and a current macOS release (macOS 15+ recommended for the pinned PyTorch version). Verify backend and BF16 support below rather than assuming availability from the hardware name. An 8B BF16 model uses roughly 16 GB just for weights; allow additional memory for loading, activations and KV cache. Prefer 32 GB or more unified memory, with 64 GB providing more headroom. Start with batch size 1. This is not a quantized or MLX implementation.

References: [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html), [Apple setup](https://developer.apple.com/metal/pytorch/), [Transformers on Apple Silicon](https://huggingface.co/docs/transformers/apple_silicon).

## 1. Get a native checkout

Use your normal checkout/copy of this project and ensure it includes the latest warmth implementation commit. The development container's Git remote is `/source`, a local path, not a public GitHub URL. No push is performed by committing here.

If you need to transfer the repository without a remote, run inside the current checkout:

```sh
git bundle create /tmp/warmth-experiments.bundle HEAD
```

Copy that bundle file to the Mac using your container/file transfer mechanism. On the Mac:

```sh
git clone /path/to/warmth-experiments.bundle warmth-experiments
cd warmth-experiments
```

A bundle includes tracked history, but not ignored model weights, vendor dependencies, activations or run data. Copy those directories separately if needed. Existing run manifests contain absolute model paths and weight mtimes, so copying a run across machines is not automatically a cache-compatible resume. Start a new Mac run unless deliberately migrating and auditing provenance. Never copy a Linux `.venv` or `.deps` as your macOS environment.

## 2. Install native Python and create an environment

If Xcode command-line tools are missing:

```sh
xcode-select --install
```

If Python 3.12 is not already installed, use Homebrew (assuming Homebrew itself is installed):

```sh
brew install python@3.12
```

From the project root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r warmth/requirements.txt
```

The requirements pin the CPU-tested Torch/Transformers versions. They are not a full transitive lockfile or a claim of MPS validation. Record the environment used for a run:

```sh
python -m pip freeze > warmth-mac-environment.txt
```

The wrapper prefers `$PYTHON_BIN`, then an active virtualenv, then the project's `.venv`, then the container interpreter if present, then `python3`. Therefore `sh warmth/run_matched.sh ...` now works outside the container. Alternatively use `python warmth/matched.py ...` directly.

## 3. Restore the Goodfire dependency

The matched pipeline imports the original pilot's runner, whose module loads Goodfire's spline utility. The full causalab environment is not needed, but its pinned source checkout is currently required:

```sh
mkdir -p vendor
git clone --branch manifold_steering https://github.com/goodfire-ai/causalab.git vendor/causalab
git -C vendor/causalab checkout 1b6f43a509d2a4ae49fd1bc4252e13889d57e38b
```

If `vendor/causalab` already exists, skip cloning and verify its revision. No upstream files are modified.

## 4. Obtain the model

Either copy the complete existing model directory from the container into `conciseness/models/llama-3.1-8b-instruct`, or download it:

```sh
python conciseness/download_model.py
```

This downloads `NousResearch/Meta-Llama-3.1-8B-Instruct`, the same repository used in the pilot. The download needs network access and roughly 16 GB of storage for weights. Inference stages use `local_files_only=True` and do not download models. An independent judge model is optional and must be obtained separately; do not reuse the rewriting model as its own independent judge.

## 5. Check MPS and run lightweight tests

```sh
python - <<'PY'
import platform
import torch
import transformers
print('Architecture:', platform.machine())
print('Torch:', torch.__version__, 'Transformers:', transformers.__version__)
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())
assert platform.machine() == 'arm64', 'Use native Python, not Rosetta'
assert torch.backends.mps.is_available(), 'Native MPS is not available'
x = torch.ones(8, device='mps', dtype=torch.bfloat16)
print('BF16 test:', (x + x).cpu())
PY
```

```sh
PYTHONPATH=warmth MPLCONFIGDIR=/tmp/warmth-mpl python -m unittest discover -s warmth -p 'test_*.py' -v
```

These tests run on CPU, including a tiny randomly initialized Llama. They test code behavior and do not establish MPS numerical equivalence or warmth efficacy. Tests requiring downloaded EPITOME data skip that dataset check when it is unavailable.

## 6. Start a small MPS run

Rather than immediately generating 300 scenarios, begin with 5 per category (20 scenarios / 60 responses):

```sh
sh warmth/run_matched.sh seed --run warmth/runs/matched_mps_smoke --per-category 5 --device mps
sh warmth/run_matched.sh draft --run warmth/runs/matched_mps_smoke --device mps
sh warmth/run_matched.sh review --run warmth/runs/matched_mps_smoke
```

Run each command separately. Fill the exported `review.csv` independently; see `MATCHED_README.md` for its rubric and optional independent-model screening. Then:

```sh
sh warmth/run_matched.sh validate --run warmth/runs/matched_mps_smoke
sh warmth/run_matched.sh collect --run warmth/runs/matched_mps_smoke --device mps --batch-size 1
sh warmth/run_matched.sh analyze --run warmth/runs/matched_mps_smoke --split validation
sh warmth/run_matched.sh check --run warmth/runs/matched_mps_smoke --device mps --method last --layer 16
sh warmth/run_matched.sh steer --run warmth/runs/matched_mps_smoke --device mps --method last --layer 16 --questions 4 --max-tokens 256
```

The small run is for plumbing and GPU checks, not research conclusions. If review rejects too many triplets, analysis correctly refuses an insufficient sample.

## 7. Full 300-scenario experiment

Use a fresh directory. Prefer imported vetted base scenarios when available:

```sh
sh warmth/run_matched.sh seed --run warmth/runs/matched_mps_v1 --input /path/to/scenarios.jsonl
```

Or generate explicitly unvalidated synthetic candidates locally:

```sh
sh warmth/run_matched.sh seed --run warmth/runs/matched_mps_v1 --device mps
```

Then follow the separate stages in `MATCHED_README.md`, always passing `--run warmth/runs/matched_mps_v1`. Add `--device mps` to `draft`, `judge`, `collect`, `check` and `steer`. `review`, `validate`, `analyze`, `select` and `score` do not perform large-model inference. Start collection at `--batch-size 1`, then try 2 after checking memory. Rerunning resumes completed records; run only one writer per directory.

If an operator is unsupported on MPS, an optional diagnostic fallback is:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 sh warmth/run_matched.sh collect --run warmth/runs/matched_mps_v1 --device mps --batch-size 1
```

Fallback does not fix all unsupported operations and can reduce speed. On memory exhaustion reduce batch size first, or deliberately choose a smaller run/context limit. Do not disable PyTorch's MPS memory safeguards. Model loading and single-sequence inference still require room for the full model.
