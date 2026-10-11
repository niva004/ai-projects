"""Exercise tests. Run from the stanford-cme295-notes/ directory:
uv run --with numpy --with pytest pytest exercises/02-models-and-tricks

Labels in the names: lecture_claim = stated in the lecture (with a timestamp or slide),
lecture_consequence = a conclusion drawn by the author of the study note,
own_addition = beyond the lecture, technical = implementation constraints.
"""
import ast
import inspect
import textwrap

import numpy as np
import pytest
import task
from task import (
    MAX_POS,
    D,
    frequencies,
    make_inputs,
    ones_curve,
    rope,
    rotation_matrix,
    score,
    sinusoidal_pe,
)

RTOL = 1e-8


@pytest.fixture
def data():
    return make_inputs(seed=0)


# --- Level 1: building blocks ---------------------------------------------------

def test_pe_shape_and_known_values():
    pe = sinusoidal_pe(8, 16)
    assert pe.shape == (8, 16)
    np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-12)  # sin(0)
    np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-12)  # cos(0)
    np.testing.assert_allclose(pe[1, 0], np.sin(1.0))
    np.testing.assert_allclose(pe[1, 1], np.cos(1.0))
    np.testing.assert_allclose(pe[3, 2], np.sin(3 * 10000 ** (-2 / 16)))
    np.testing.assert_allclose(pe[3, 3], np.cos(3 * 10000 ** (-2 / 16)))


def test_rotation_matrix_shape_and_block_structure():
    R = rotation_matrix(5, 8)
    th = frequencies(8)[1]
    assert R.shape == (8, 8)
    np.testing.assert_allclose(R[2:4, 2:4], [[np.cos(5 * th), -np.sin(5 * th)],
                                              [np.sin(5 * th), np.cos(5 * th)]])
    assert np.all(R[0:2, 2:] == 0.0)  # zeros outside the blocks


def test_at_position_zero_rope_and_rotation_matrix_are_identity(data):
    x_m, *_ = data
    np.testing.assert_allclose(rope(x_m, 0), x_m)
    np.testing.assert_allclose(rotation_matrix(0, D), np.eye(D), atol=1e-12)


# --- Test 1: consequence of R being a rotation (31:37; 33:17) --------------------

@pytest.mark.parametrize("m", [1, 7, 100, 1024])
def test_lecture_consequence_31_37_rope_preserves_the_norm(data, m):
    x_m, *_ = data
    np.testing.assert_allclose(np.linalg.norm(rope(x_m, m)), np.linalg.norm(x_m), rtol=RTOL)


@pytest.mark.parametrize("m", [1, 7, 100, 1024])
def test_lecture_consequence_33_17_rope_equals_the_explicit_rotation_matrix(data, m):
    x_m, *_ = data
    np.testing.assert_allclose(rope(x_m, m), rotation_matrix(m, D) @ x_m, rtol=RTOL, atol=1e-10)


def test_lecture_consequence_rotation_matrix_is_orthogonal():
    R = rotation_matrix(37, D)
    np.testing.assert_allclose(R @ R.T, np.eye(D), atol=1e-10)


def test_rope_with_an_array_of_positions_rotates_each_row_by_its_own_angle(data):
    x_m, x_n, *_ = data
    X = np.stack([x_m, x_n, x_m])
    pos = np.array([2, 40, 900])
    out = rope(X, pos)
    assert out.shape == X.shape
    for row, p, x in zip(out, pos, X):
        np.testing.assert_allclose(row, rope(x, int(p)), rtol=RTOL, atol=1e-10)


# --- Test 2: claim from slide 33 --------------------------------------------------

@pytest.mark.parametrize("m, n", [(100, 107), (500, 507)])
def test_lecture_claim_slide_33_score_depends_only_on_position_difference(data, m, n):
    x_m, x_n, W_q, W_k = data
    ref = score(x_m, 3, x_n, 10, W_q, W_k)
    np.testing.assert_allclose(score(x_m, m, x_n, n, W_q, W_k), ref, rtol=RTOL, atol=1e-10)


def test_lecture_claim_slide_33_a_different_difference_gives_a_different_score(data):
    x_m, x_n, W_q, W_k = data
    a = score(x_m, 3, x_n, 10, W_q, W_k)
    b = score(x_m, 3, x_n, 50, W_q, W_k)
    assert not np.isclose(a, b, rtol=1e-3)


def test_lecture_claim_slide_33_formula_with_matrix_R_n_minus_m(data):
    # q_m k_n^T = x_m W_q R_{theta,n-m} W_k^T x_n^T  (slide 33)
    x_m, x_n, W_q, W_k = data
    for m, n in [(3, 10), (50, 20)]:  # also n < m (negative angle)
        expected = (x_m @ W_q) @ rotation_matrix(n - m, D) @ (x_n @ W_k)
        np.testing.assert_allclose(score(x_m, m, x_n, n, W_q, W_k), expected,
                                   rtol=RTOL, atol=1e-10)


# --- Test 3: claims from slide 21 and from 21:45 ----------------------------------

def test_lecture_claim_slide_21_pe_dot_products_are_constant_along_diagonals():
    pe = sinusoidal_pe(MAX_POS, D)
    gram = pe @ pe.T
    for k in (0, 1, 7, 100, 1000):
        diag = np.diagonal(gram, offset=k)
        np.testing.assert_allclose(diag, diag[0], rtol=RTOL, atol=RTOL * D)


def test_lecture_claim_21_45_row_maximum_is_on_the_main_diagonal():
    pe = sinusoidal_pe(MAX_POS, D)
    gram = pe @ pe.T
    np.testing.assert_array_equal(gram.argmax(axis=1), np.arange(MAX_POS))
    np.testing.assert_allclose(np.diag(gram), D / 2, rtol=RTOL)  # sin^2 + cos^2 = 1 in d/2 pairs


# --- Test 4: own addition ---------------------------------------------------------

DELTAS = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])


def test_own_addition_ones_curve_has_its_maximum_at_delta_zero():
    curve = ones_curve(DELTAS)
    print("\nq_0 . k_delta after RoPE (q = k = ones):")
    for delta, value in zip(DELTAS, curve):
        print(f"  delta={delta:5d}  {value:8.3f}")
    assert curve.shape == DELTAS.shape
    assert curve[0] == pytest.approx(D, rel=RTOL)
    assert np.all(curve[1:] < curve[0])


def test_own_addition_ones_curve_matches_the_2_sum_cos_formula():
    # (1,1) rotated by angle a and dotted with (1,1): 2 cos(a) per pair
    expected = np.array([2 * np.cos(d * frequencies(D)).sum() for d in DELTAS])
    np.testing.assert_allclose(ones_curve(DELTAS), expected, rtol=RTOL, atol=1e-9)


# --- Test 5: no loops ---------------------------------------------------------------

BANNED = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@pytest.mark.parametrize("func", [task.rope, task.sinusoidal_pe])
def test_technical_no_loops_in_rope_and_sinusoidal_pe(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    assert not [n for n in ast.walk(tree) if isinstance(n, BANNED)]


def test_technical_rope_does_not_build_the_rotation_matrix():
    tree = ast.parse(textwrap.dedent(inspect.getsource(task.rope)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "rotation_matrix" not in names
