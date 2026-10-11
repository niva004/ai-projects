"""Exercise tests. Run: uv run --with numpy --with pytest pytest exercises/01-transformer"""
import ast
import inspect
import textwrap

import numpy as np
import pytest
import task
from task import (
    attention,
    make_inputs,
    mean_max_weight,
    multi_head,
    score_std,
    sinusoidal_positions,
    softmax,
)


@pytest.fixture
def data():
    return make_inputs(seed=0)


# --- Level 1: building blocks -------------------------------------------------

def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(1).normal(size=(5, 7))
    np.testing.assert_allclose(softmax(x).sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_is_robust_to_large_numbers():
    out = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)


def test_softmax_gives_zero_for_minus_infinity():
    out = softmax(np.array([[0.0, -np.inf, 0.0]]))
    np.testing.assert_allclose(out, [[0.5, 0.0, 0.5]])


def test_positions_known_values():
    pe = sinusoidal_positions(8, 16)
    assert pe.shape == (8, 16)
    np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-12)  # sin(0)
    np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-12)  # cos(0)
    np.testing.assert_allclose(pe[1, 0], np.sin(1.0))
    np.testing.assert_allclose(pe[1, 1], np.cos(1.0))
    np.testing.assert_allclose(pe[3, 2], np.sin(3.0 / 10000 ** (2 / 16)))


def test_positions_every_position_is_different():
    pe = sinusoidal_positions(8, 16)
    assert len({tuple(np.round(row, 8)) for row in pe}) == 8


# --- Criterion 1: shapes and weights as a distribution -------------------------

def test_attention_shapes():
    rng = np.random.default_rng(2)
    Q, K, V = rng.normal(size=(8, 4)), rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    out, w = attention(Q, K, V)
    assert out.shape == (8, 4)
    assert w.shape == (8, 8)


def test_attention_matches_hand_computed_formula():
    # 2 tokens, d_k = 4 -> divide by 2
    Q = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0]])
    K = np.array([[2.0, 0, 0, 0], [0, 0, 0, 0]])
    V = np.array([[1.0, 0], [0, 1.0]])
    out, w = attention(Q, K, V)
    e = np.exp(2.0)
    np.testing.assert_allclose(w[0], [e / (e + 1), 1 / (e + 1)])
    np.testing.assert_allclose(w[1], [0.5, 0.5])
    np.testing.assert_allclose(out, w @ V)


def test_zero_queries_give_equal_weights():
    K = np.random.default_rng(3).normal(size=(5, 4))
    _, w = attention(np.zeros((5, 4)), K, K)
    np.testing.assert_allclose(w, np.full((5, 5), 0.2))


@pytest.mark.parametrize("causal", [False, True])
def test_weights_sum_to_one(data, causal):
    X, W_Q, W_K, W_V, _ = data
    for i in range(W_Q.shape[0]):
        _, w = attention(X @ W_Q[i], X @ W_K[i], X @ W_V[i], causal=causal)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(w >= 0)


def test_multi_head_shape(data):
    X, *W = data
    assert multi_head(X, *W).shape == (8, 16)


def test_multi_head_matches_reference_implementation(data):
    X, W_Q, W_K, W_V, W_O = data
    heads = []
    for i in range(4):
        Q, K, V = X @ W_Q[i], X @ W_K[i], X @ W_V[i]
        s = Q @ K.T / np.sqrt(4)
        w = np.exp(s - s.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        heads.append(w @ V)
    expected = np.hstack(heads) @ W_O
    np.testing.assert_allclose(multi_head(X, W_Q, W_K, W_V, W_O), expected, atol=1e-8)


# --- Criterion 2: causal mask ---------------------------------------------------

def test_mask_zeroes_weights_above_the_diagonal(data):
    X, W_Q, W_K, W_V, _ = data
    _, w = attention(X @ W_Q[0], X @ W_K[0], X @ W_V[0], causal=True)
    assert np.all(np.triu(w, k=1) == 0.0)
    assert w[0, 0] == pytest.approx(1.0)  # the first token sees only itself


def test_without_mask_weights_above_the_diagonal_are_positive(data):
    X, W_Q, W_K, W_V, _ = data
    _, w = attention(X @ W_Q[0], X @ W_K[0], X @ W_V[0], causal=False)
    assert np.all(w[np.triu_indices_from(w, k=1)] > 0)


def test_changing_the_last_token_does_not_change_earlier_outputs(data):
    X, *W = data
    X2 = X.copy()
    X2[-1] = np.random.default_rng(9).normal(size=X.shape[1])
    a, b = multi_head(X, *W, causal=True), multi_head(X2, *W, causal=True)
    np.testing.assert_allclose(a[:-1], b[:-1], atol=1e-10)
    assert not np.allclose(a[-1], b[-1])


def test_without_mask_the_last_token_affects_earlier_outputs(data):
    X, *W = data
    X2 = X.copy()
    X2[-1] = np.random.default_rng(9).normal(size=X.shape[1])
    a, b = multi_head(X, *W), multi_head(X2, *W)
    assert not np.allclose(a[:-1], b[:-1])


# --- Criterion 3: token order ---------------------------------------------------

def test_lecture_consequence_1_19_42_permutation_without_positions_causal_false(data):
    # Consequence of the lecture (1:19:42, no sense of order); only causal=False,
    # because the causal mask itself introduces a dependence on order.
    X, *W = data
    perm = np.random.default_rng(4).permutation(len(X))
    np.testing.assert_allclose(
        multi_head(X[perm], *W, causal=False),
        multi_head(X, *W, causal=False)[perm],
        atol=1e-10,
    )


def test_lecture_consequence_with_positional_encoding_order_matters(data):
    X, *W = data
    pe = sinusoidal_positions(*X.shape)
    perm = np.random.default_rng(4).permutation(len(X))
    permuted_in = multi_head(X[perm] + pe, *W, causal=False)
    permuted_out = multi_head(X + pe, *W, causal=False)[perm]
    assert not np.allclose(permuted_in, permuted_out, atol=1e-3)


# --- Criterion 4: scaling by sqrt(d_k) -------------------------------------------

def test_lecture_consequence_1_33_51_std_grows_like_sqrt_dk():
    # The lecture (1:33:51) says only that the dot products grow with the dimension;
    # the assertion s512 ≈ √512 is our own calculation (a sum of d_k products of
    # independent N(0,1) variables).
    s4, s512 = score_std(4, scale=False), score_std(512, scale=False)
    print(f"\nstd without scaling: d_k=4 -> {s4:.2f}, d_k=512 -> {s512:.2f}")
    assert s4 == pytest.approx(2.0, rel=0.15)
    assert s512 == pytest.approx(np.sqrt(512), rel=0.15)
    assert s512 > 8 * s4


def test_lecture_consequence_1_33_51_after_dividing_by_sqrt_dk_std_is_constant():
    s4, s512 = score_std(4, scale=True), score_std(512, scale=True)
    assert s4 == pytest.approx(1.0, rel=0.15)
    assert s512 == pytest.approx(1.0, rel=0.15)


def test_mean_max_weight_is_deterministic():
    assert mean_max_weight(4, True) == mean_max_weight(4, True)


def test_own_addition_without_scaling_softmax_becomes_hard():
    without = mean_max_weight(512, scale=False)
    with_scaling = mean_max_weight(512, scale=True)
    print(f"\nd_k=512: without scaling {without:.3f}, with scaling {with_scaling:.3f}")
    assert without > 0.9
    assert with_scaling < 0.5
    assert without - with_scaling > 0.4


def test_with_scaling_the_result_barely_depends_on_dk():
    assert abs(mean_max_weight(4, True) - mean_max_weight(512, True)) < 0.1


# --- Criterion 5: no loops over tokens --------------------------------------------

@pytest.mark.parametrize("func", [task.attention, task.softmax])
def test_technical_no_loops_in_attention_and_softmax(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    banned = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    assert not [n for n in ast.walk(tree) if isinstance(n, banned)]
