"""Baseline profiling of open weight model inference.

Measures greedy decoding throughput and peak GPU memory with the KV cache
on and off, so later kernel and cache experiments have a number to beat.

Usage:
    python benchmarks/baseline_profile.py --model gpt2 --new-tokens 128
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def timed_generate(model, tokenizer, inputs, new_tokens, use_cache):
    on_gpu = model.device.type == "cuda"
    if on_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    if on_gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = out.shape[1] - inputs["input_ids"].shape[1]
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if on_gpu else float("nan")
    return generated / elapsed, peak_mb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="The key value cache in a transformer")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # Warm up so one time setup cost does not land in the first measurement.
    timed_generate(model, tokenizer, inputs, 8, True)

    print(f"model={args.model} device={device} new_tokens={args.new_tokens}")
    for use_cache in (True, False):
        runs = [
            timed_generate(model, tokenizer, inputs, args.new_tokens, use_cache)
            for _ in range(args.repeats)
        ]
        best_tps = max(tps for tps, _ in runs)
        peak_mb = max(mb for _, mb in runs)
        print(f"use_cache={use_cache!s:5}  {best_tps:8.1f} tok/s  peak {peak_mb:8.1f} MB")


if __name__ == "__main__":
    main()
