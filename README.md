# Redline Inference

GPU kernels and KV cache optimizations for open weight model inference, plus a red teaming harness that stress tests models and agents for security failures.

## Status

Prototype, in active development. This repo currently holds the plan and early experiments. Code lands here as each piece becomes runnable.

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
