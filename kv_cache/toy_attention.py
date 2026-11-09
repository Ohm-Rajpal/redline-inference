"""Toy single head causal attention decode, with and without a KV cache.

No GPU or framework needed. Shows that the cached path produces the same
outputs as recomputing the full prefix each step, and how the cost per step
grows without the cache.

Usage:
    python kv_cache/toy_attention.py --steps 256 --dim 256
"""
import argparse
import time

import numpy as np


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def decode_no_cache(tokens, wq, wk, wv):
    """Recompute keys and values for the whole prefix at every step."""
    outputs = []
    scale = np.sqrt(wq.shape[1])
    for t in range(1, len(tokens) + 1):
        prefix = tokens[:t]
        q = prefix[-1] @ wq
        k = prefix @ wk
        v = prefix @ wv
        outputs.append(softmax(q @ k.T / scale) @ v)
    return np.stack(outputs)


def decode_with_cache(tokens, wq, wk, wv):
    """Project each token once and append to preallocated K and V buffers."""
    n, d = len(tokens), wk.shape[1]
    k_cache = np.empty((n, d))
    v_cache = np.empty((n, d))
    outputs = []
    scale = np.sqrt(wq.shape[1])
    for t, x in enumerate(tokens):
        k_cache[t] = x @ wk
        v_cache[t] = x @ wv
        q = x @ wq
        outputs.append(softmax(q @ k_cache[: t + 1].T / scale) @ v_cache[: t + 1])
    return np.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((args.steps, args.dim))
    wq, wk, wv = (rng.standard_normal((args.dim, args.dim)) / np.sqrt(args.dim) for _ in range(3))

    start = time.perf_counter()
    slow = decode_no_cache(tokens, wq, wk, wv)
    t_slow = time.perf_counter() - start

    start = time.perf_counter()
    fast = decode_with_cache(tokens, wq, wk, wv)
    t_fast = time.perf_counter() - start

    assert np.allclose(slow, fast), "cached and uncached outputs differ"
    print(f"steps={args.steps} dim={args.dim}")
    print(f"no cache    {t_slow * 1e3:8.1f} ms")
    print(f"with cache  {t_fast * 1e3:8.1f} ms  ({t_slow / t_fast:.1f}x faster)")


if __name__ == "__main__":
    main()
