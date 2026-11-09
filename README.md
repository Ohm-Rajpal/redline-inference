# Redline Inference

GPU kernels and KV cache optimizations for open weight model inference, plus a red teaming harness that stress tests models and agents for security failures.

For teams running open weight models in production who need faster, cheaper inference and evidence that their models and agents hold up against attacks.

## Status

Prototype, in active development. The repo holds a baseline profiling script, a toy KV cache experiment, and a first stub of the red teaming harness. More lands here as each piece becomes runnable.

## What is in the repo

| Path | What it does |
| --- | --- |
| `benchmarks/baseline_profile.py` | Measures decoding throughput and peak GPU memory for an open weight model with the KV cache on and off |
| `kv_cache/toy_attention.py` | Framework free attention decode that checks cached and uncached outputs match and times both |
| `harness/harness.py` | Harness v0 stub that runs prompt injection cases against a pluggable target and reports canary leaks |

Quick start

```
pip install -r requirements.txt
python kv_cache/toy_attention.py
python harness/harness.py
python benchmarks/baseline_profile.py --model gpt2
```

## What I am working on now

* **Custom GPU kernels** for core inference operations (attention, matmul, normalization), written and profiled against library baselines
* **KV cache experiments** covering memory layout, reuse across requests, and eviction, measured on open weight models
* **Red teaming harness** (early) that runs adversarial agents against self hosted open weight models and records security failures such as prompt injection and unsafe tool use

## Roadmap

1. Baseline profiling of open weight model inference on a single GPU
2. Hand written kernels for the hottest operations, with benchmarks against the baseline
3. KV cache variants with latency and memory comparisons
4. Red teaming harness v0 with a small attack suite and a reproducible report format
5. Security flavored inference work, for example checking whether cache sharing across requests leaks information

## Why this needs cloud GPUs

Development happens on a laptop with an 8 GB RTX 4070, which limits what can be measured.

* Profiling kernels on data center class GPUs, where the performance behavior differs from a laptop card
* Running larger open weight models that do not fit in 8 GB of VRAM
* Training and fine tuning experiments
* Running red teaming agents against self hosted targets at useful scale

## About

Built by Ohm Rajpal, EECS at UC Berkeley. Interests are AI security, red teaming, and inference systems.
