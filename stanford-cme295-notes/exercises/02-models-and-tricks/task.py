"""Exercise for lecture 2 (CME 295): sinusoidal position embeddings and RoPE in numpy.

Shape notation:
    d        vector dimension (64, the exercise's choice), even; the vector consists of d/2 pairs (2i, 2i+1)
    n_pos    number of positions (up to 1024)
    omega_i  = 10000^(-2i/d), i = 0..d/2-1 in the code (formula: slide 16; range given orally, 41:11); in RoPE theta_i = omega_i (41:11)

RoPE convention (slide 30 writes the row form q R^T; here the column form): the vector x at position m is rotated in
every pair (x[2i], x[2i+1]) by the angle m * theta_i, counter-clockwise:
    x'[2i]   = x[2i] cos - x[2i+1] sin
    x'[2i+1] = x[2i] sin + x[2i+1] cos
This is R_{theta,m} @ x (x as a column), i.e. x @ R_{theta,m}^T (x as a row).
"""
import numpy as np

D = 64
MAX_POS = 1024


def frequencies(d=D):
    """READY-MADE, do not change. omega_i = 10000^(-2i/d), shape (d/2,)."""
    return 10000.0 ** (-2.0 * np.arange(d // 2) / d)


def make_inputs(seed=0, d=D):
    """READY-MADE, do not change. Returns (x_m, x_n, W_q, W_k).

    x_m, x_n: (d,) random vectors (embeddings of two tokens, without position)
    W_q, W_k: (d, d) random projections, scaled by 1/sqrt(d)
    """
    rng = np.random.default_rng(seed)
    x_m = rng.normal(size=d)
    x_n = rng.normal(size=d)
    W_q = rng.normal(size=(d, d)) / np.sqrt(d)
    W_k = rng.normal(size=(d, d)) / np.sqrt(d)
    return x_m, x_n, W_q, W_k


def sinusoidal_pe(n_pos, d):
    """Sinusoidal position embeddings (slide 15), shape (n_pos, d).

    PE[m, 2i]   = sin(omega_i * m)
    PE[m, 2i+1] = cos(omega_i * m)
    No loops over positions or coordinates (use broadcasting and frequencies(d)).
    """
    raise NotImplementedError


def rotation_matrix(m, d):
    """Explicit matrix R_{theta,m}, shape (d, d) (slide 32).

    Block-diagonal: block i (rows/columns 2i, 2i+1) is
        [[cos(m*theta_i), -sin(m*theta_i)],
         [sin(m*theta_i),  cos(m*theta_i)]]
    where theta_i = frequencies(d)[i]. A loop over blocks is allowed here.
    """
    raise NotImplementedError


def rope(x, m):
    """RoPE: rotate coordinate pairs of x by the angles m * theta_i, without a d x d matrix.

    x: (..., d)
    m: an integer or an array of positions of shape (...) (e.g. np.arange(n)
       for x of shape (n, d)); each position rotates its own vector.
    Returns an array of shape x.shape. The result must equal R_{theta,m} @ x.

    No loops; you must not use rotation_matrix.
    """
    raise NotImplementedError


def score(x_m, m, x_n, n, W_q, W_k):
    """Attention score q_m . k_n after RoPE (slide 33), a number (float).

    q_m = x_m @ W_q, k_n = x_n @ W_k; then rotate q by position m, k by position n
    and take the dot product (no division by sqrt(d_k), no softmax).
    """
    raise NotImplementedError


def ones_curve(deltas, d=D):
    """Own addition: q_0 . k_delta after RoPE for q = k = the vector of ones (d,).

    deltas: array (k,) of distances. Returns (k,) values: q at position 0,
    k at position delta. For delta = 0 the value is d (squared norm of the ones).
    """
    raise NotImplementedError


def report():
    """Prints the curve from test 4 and the score for a few position pairs."""
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
