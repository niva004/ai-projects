"""Reference solution for the lecture 2 exercise (CME 295)."""
import numpy as np

D = 64
MAX_POS = 1024


def frequencies(d=D):
    return 10000.0 ** (-2.0 * np.arange(d // 2) / d)


def make_inputs(seed=0, d=D):
    rng = np.random.default_rng(seed)
    x_m = rng.normal(size=d)
    x_n = rng.normal(size=d)
    W_q = rng.normal(size=(d, d)) / np.sqrt(d)
    W_k = rng.normal(size=(d, d)) / np.sqrt(d)
    return x_m, x_n, W_q, W_k


def sinusoidal_pe(n_pos, d):
    angles = np.arange(n_pos)[:, None] * frequencies(d)[None, :]
    pe = np.empty((n_pos, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def rotation_matrix(m, d):
    R = np.zeros((d, d))
    for i, theta in enumerate(frequencies(d)):
        c, s = np.cos(m * theta), np.sin(m * theta)
        R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[c, -s], [s, c]]
    return R


def rope(x, m):
    x = np.asarray(x, dtype=float)
    angles = np.asarray(m)[..., None] * frequencies(x.shape[-1])
    c, s = np.cos(angles), np.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = even * s + odd * c
    return out


def score(x_m, m, x_n, n, W_q, W_k):
    return float(rope(x_m @ W_q, m) @ rope(x_n @ W_k, n))


def ones_curve(deltas, d=D):
    deltas = np.asarray(deltas)
    ones = np.ones(d)
    keys = rope(np.broadcast_to(ones, (len(deltas), d)), deltas)
    return keys @ rope(ones, 0)


def report():
    x_m, x_n, W_q, W_k = make_inputs()
    print("score for a constant position difference of 7 and for a difference of 47:")
    for m, n in [(3, 10), (100, 107), (500, 507), (3, 50)]:
        print(f"  m={m:4d} n={n:4d}  score = {score(x_m, m, x_n, n, W_q, W_k):.6f}")
    deltas = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    print("q_0 . k_delta after RoPE for q = k = ones:")
    for delta, value in zip(deltas, ones_curve(deltas)):
        print(f"  delta={delta:5d}  {value:8.3f}")


if __name__ == "__main__":
    report()
