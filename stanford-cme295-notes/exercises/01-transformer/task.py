"""Exercise for lecture 1 (CME 295): attention in pure numpy.

Shape notation:
    n        number of tokens in the sequence
    d_model  embedding dimension (16)
    h        number of heads (4)
    d_k      dimension of one head (4); here d_k = d_model / h
"""
import numpy as np

N_TOKENS = 8
D_MODEL = 16
N_HEADS = 4
D_K = 4


def make_inputs(seed=0, n=N_TOKENS, d_model=D_MODEL, h=N_HEADS, d_k=D_K):
    """READY-MADE, do not change. Returns (X, W_Q, W_K, W_V, W_O).

    X:   (n, d_model)
    W_Q, W_K, W_V: (h, d_model, d_k)  -- one matrix per head
    W_O: (h * d_k, d_model)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_model))
    W_Q = rng.normal(size=(h, d_model, d_k)) / np.sqrt(d_model)
    W_K = rng.normal(size=(h, d_model, d_k)) / np.sqrt(d_model)
    W_V = rng.normal(size=(h, d_model, d_k)) / np.sqrt(d_model)
    W_O = rng.normal(size=(h * d_k, d_model)) / np.sqrt(h * d_k)
    return X, W_Q, W_K, W_V, W_O


def softmax(x):
    """Softmax over the last axis, safe against overflow.

    x: array of any shape (..., m). Returns an array of the same shape in which
    every vector along the last axis sums to 1.
    Hint: softmax(x) = softmax(x - max(x)), and exp(-inf) = 0.
    """
    raise NotImplementedError


def attention(Q, K, V, causal=False, scale=True):
    """Scaled dot-product attention.

    Q: (n, d_k), K: (n, d_k), V: (n, d_v)
    causal: if True, token i must not look at tokens j > i
    scale:  if True, the QK^T scores are divided by sqrt(d_k); False only for experiment 4

    Returns (output, weights):
        output:  (n, d_v)
        weights: (n, n), weights[i, j] = how much token i takes from token j; rows sum to 1

    No loops over tokens, matrix operations only.
    """
    raise NotImplementedError


def multi_head(X, W_Q, W_K, W_V, W_O, causal=False):
    """Multi-head attention.

    X: (n, d_model); W_Q, W_K, W_V: (h, d_model, d_k); W_O: (h * d_k, d_model)
    For each head: Q = X @ W_Q[i] etc., then attention. Concatenate the head
    outputs (n, d_k) along the feature axis to (n, h * d_k) and multiply by W_O.

    Returns (n, d_model). A loop over heads is allowed (over tokens it is not).
    """
    raise NotImplementedError


def sinusoidal_positions(n, d_model):
    """Sinusoidal positional encoding, shape (n, d_model).

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
    d_model is assumed to be even.
    """
    raise NotImplementedError


def score_std(d_k, scale, n=16, trials=200, seed=0):
    """Experiment 4 (from the lecture, 1:33:51): standard deviation of the dot products.

    In each trial draw Q and K of shape (n, d_k) with i.i.d. N(0, 1) entries
    (one generator np.random.default_rng(seed) for the whole run). Compute
    Q @ K.T; if scale=True, divide by sqrt(d_k). Return the standard deviation
    of all elements from all trials as a float.
    Expectation: about sqrt(d_k) without scaling, about 1 with scaling.
    """
    raise NotImplementedError


def mean_max_weight(d_k, scale, n=16, trials=200, seed=0):
    """Own addition, beyond the lecture: observing the effect of not scaling in the softmax.

    Mean (over trials and rows) of the largest softmax weight.

    In each trial draw Q and K ~ N(0, 1) of shape (n, d_k) (one generator
    np.random.default_rng(seed) for the whole run), compute the weights with
    attention(..., scale=scale) and take the maximum of each row. Return the
    mean as a float. Close to 1 = the softmax is almost "hard", the gradient vanishes.
    """
    raise NotImplementedError


def report():
    """Prints a report: the weights of one head and experiment 4."""
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
