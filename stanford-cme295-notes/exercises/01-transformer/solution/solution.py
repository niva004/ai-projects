"""Reference solution for the lecture 1 exercise (CME 295)."""
import numpy as np

N_TOKENS = 8
D_MODEL = 16
N_HEADS = 4
D_K = 4


def make_inputs(seed=0, n=N_TOKENS, d_model=D_MODEL, h=N_HEADS, d_k=D_K):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_model))
    W_Q = rng.normal(size=(h, d_model, d_k)) / np.sqrt(d_model)
    W_K = rng.normal(size=(h, d_model, d_k)) / np.sqrt(d_model)
    W_V = rng.normal(size=(h, d_model, d_k)) / np.sqrt(d_model)
    W_O = rng.normal(size=(h * d_k, d_model)) / np.sqrt(h * d_k)
    return X, W_Q, W_K, W_V, W_O


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(Q, K, V, causal=False, scale=True):
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(K.shape[-1])
    if causal:
        n = scores.shape[0]
        future = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
    weights = softmax(scores)
    return weights @ V, weights


def multi_head(X, W_Q, W_K, W_V, W_O, causal=False):
    heads = [
        attention(X @ wq, X @ wk, X @ wv, causal=causal)[0]
        for wq, wk, wv in zip(W_Q, W_K, W_V)
    ]
    return np.concatenate(heads, axis=-1) @ W_O


def sinusoidal_positions(n, d_model):
    pos = np.arange(n)[:, None]
    two_i = np.arange(0, d_model, 2)[None, :]
    angles = pos / np.power(10000.0, two_i / d_model)
    pe = np.zeros((n, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def score_std(d_k, scale, n=16, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(trials):
        Q = rng.normal(size=(n, d_k))
        K = rng.normal(size=(n, d_k))
        s = Q @ K.T
        scores.append(s / np.sqrt(d_k) if scale else s)
    return float(np.std(np.stack(scores)))


def mean_max_weight(d_k, scale, n=16, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(trials):
        Q = rng.normal(size=(n, d_k))
        K = rng.normal(size=(n, d_k))
        _, w = attention(Q, K, K, scale=scale)
        maxima.append(w.max(axis=1))
    return float(np.mean(maxima))


def report():
    X, W_Q, W_K, W_V, W_O = make_inputs()
    Q, K, V = X @ W_Q[0], X @ W_K[0], X @ W_V[0]
    _, weights = attention(Q, K, V, causal=True)
    np.set_printoptions(precision=2, suppress=True)
    print("Weights of head 0 (causal=True):")
    print(weights)
    for d_k in (4, 512):
        for scale in (True, False):
            std = score_std(d_k, scale)
            print(f"d_k={d_k:4d} scale={scale!s:5}  std of dot products = {std:.3f}")
    print("Own addition, beyond the lecture:")
    for d_k in (4, 512):
        for scale in (True, False):
            value = mean_max_weight(d_k, scale)
            print(f"d_k={d_k:4d} scale={scale!s:5}  mean max weight = {value:.3f}")


if __name__ == "__main__":
    report()
