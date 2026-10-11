# Exercise: relative position, sinusoids and RoPE in numpy

Time: 30-60 minutes. You need Python and numpy. Knowing attention from lecture 1 is enough; you get the rest below.

## Goal

Implement sinusoidal position embeddings and RoPE (rotary position embeddings), then check with experiments that both encode relative position:

1. RoPE is a rotation, so it preserves the norm and equals an explicit rotation matrix (consequence: 31:37, 33:17),
2. the attention score $q_m k_n^T$ after RoPE depends only on $n-m$ (claim of the lecture, slide 33),
3. the dot products of sinusoidal embeddings depend only on $m-n$ and are largest for $m=n$ (claim of the lecture, slide 21 and 21:45),
4. own addition, beyond the lecture: the curve $q_0 \cdot k_\Delta$ after RoPE for vectors of ones.

What we left out: softmax, the mask, multiple heads, training and all other ways of encoding position (T5, ALiBi). We work on a single pair of vectors, not a whole attention layer. The lecture skipped the math of RoPE for lack of time (38:04), so the proof of the property is replaced here by a numerical check.

## Shapes and constants

| Symbol | Shape / value | Meaning |
|---|---|---|
| $d$ | 64 | the exercise's choice, not the lecture's; vector dimension; $d/2 = 32$ coordinate pairs $(2i, 2i+1)$ |
| positions | $0, \dots, 1024$ | the exercise's choice, not the lecture's |
| $\omega_i = 10000^{-2i/d}$ | $(d/2,)$ | frequencies (formula from slide 16, which does not give the range of $i$); in the code $i = 0, \dots, d/2-1$ (zero-based), while the lecture says orally $i$ from 1 to $d/2$ (41:11); in RoPE $\theta_i = \omega_i$ (orally, 41:11); ready-made function `frequencies(d)` |
| $PE$ | $(n_{pos}, d)$ | $PE_{m,2i} = \sin(\omega_i m)$, $PE_{m,2i+1} = \cos(\omega_i m)$ (slide 15) |
| $R_{\theta,m}$ | $(d, d)$ | block-diagonal; block $i$ is a rotation by the angle $m\theta_i$ (slides 30, 32) |
| $x_m, x_n$ | $(d,)$ | random token embeddings (seed 0, `make_inputs`) |
| $W_q, W_k$ | $(d, d)$ | random projections |
| $q_m = x_m W_q$, $k_n = x_n W_k$ | $(d,)$ | query and key before rotation |

Rotation block: $\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}$.

Convention (stated once): the lecture (slide 30) writes the rotation for a row vector as $q\,R_{\theta,m}^T$. The code uses the equivalent column form: `rope(x, m)` must give $R_{\theta,m}\,x$, so in a pair $(a, b)$ the result is $(a\cos - b\sin,\; a\sin + b\cos)$ for the angle $m\theta_i$. This way the product depends on $R_{\theta,n-m}$: $q_m k_n^T = x_m W_q R_{\theta,n-m} W_k^T x_n^T$ (slide 33).

## What to build

File `task.py`: fill in `sinusoidal_pe`, `rotation_matrix`, `rope`, `score`, `ones_curve`. `frequencies`, `make_inputs` and `report` are ready.

- `sinusoidal_pe(n_pos, d)` returns $(n_{pos}, d)$.
- `rotation_matrix(m, d)` returns the explicit $(d, d)$ matrix (a loop over blocks is allowed).
- `rope(x, m)`: rotation of coordinate pairs, without a $d \times d$ matrix. `x` has shape $(\dots, d)$, `m` is a number or an array of positions of shape $(\dots)$.
- `score(x_m, m, x_n, n, W_q, W_k)` returns the number $q_m \cdot k_n$ after rotation. The signature includes $W_q, W_k$ so that the function does not depend on global state.
- `ones_curve(deltas)` returns $q_0 \cdot k_\Delta$ after RoPE for $q = k$ = the vector of ones.
- `python task.py` prints the `score` results for four position pairs and the curve from test 4.

## Passing criteria

Relative tolerance $10^{-8}$. All tests pass:

1. Consequence of $R_{\theta,m}$ being a rotation (31:37; 33:17): `rope` preserves the norm, gives the same as `rotation_matrix(m, d) @ x`, and the matrix itself is orthogonal.
2. Claim of the lecture (slide 33): `score` is equal for $(m,n) = (3,10), (100,107), (500,507)$, different for $(3,50)$, and consistent with the formula $x_m W_q R_{\theta,n-m} W_k^T x_n^T$.
3. Claim of the lecture (slide 21 and 21:45): $PE\,PE^T$ is constant along every diagonal (1024 positions), and the maximum of each row lies on the main diagonal.
4. Own addition, beyond the lecture: for $q = k$ = ones, `ones_curve` for $\Delta = 0, 1, 2, 4, \dots, 1024$ (positions $0, \dots, 1024$, because $\Delta = 1024$ needs position 1024) has its largest value at $\Delta = 0$ (equal to $d$) and agrees with the formula $2\sum_i \cos(\Delta\omega_i)$. You can compare the curve with slide 35, but that slide shows an upper bound, not this quantity.
5. Technical: `rope` and `sinusoidal_pe` contain no loops or comprehensions, and `rope` does not use `rotation_matrix`.

## How to run

From the repository directory `stanford-cme295-notes/`, run each exercise separately (both test files are named `test_task.py`):

```
uv run --with numpy --with pytest pytest exercises/02-models-and-tricks
uv run --with numpy python exercises/02-models-and-tricks/task.py
```

or, without uv:

```
pip install numpy pytest && pytest exercises/02-models-and-tricks
```

The reference solution is hidden in `solution/solution.py`. To check it, copy it over `task.py` (or into a temporary directory next to `test_task.py` as `task.py`).

## Hints

Read them one at a time, only when you are stuck.

<details>
<summary>Hint 1: sinusoidal_pe and rotation_matrix</summary>

Angles: `np.arange(n_pos)[:, None] * frequencies(d)[None, :]` has shape $(n_{pos}, d/2)$. Put the sines into columns `0::2` and the cosines into `1::2`. In `rotation_matrix`, for block `i` insert `[[c, -s], [s, c]]` into `R[2*i:2*i+2, 2*i:2*i+2]`.
</details>

<details>
<summary>Hint 2: rope without a matrix</summary>

Split `even = x[..., 0::2]` and `odd = x[..., 1::2]`. Angles: `np.asarray(m)[..., None] * frequencies(x.shape[-1])`; this works for both a number and an array of positions. Result: `even*cos - odd*sin` goes to the even columns, `even*sin + odd*cos` to the odd columns.
</details>

<details>
<summary>Hint 3: score and ones_curve</summary>

`score`: `rope(x_m @ W_q, m) @ rope(x_n @ W_k, n)`. For `ones_curve` build a `(len(deltas), d)` matrix of ones (`np.broadcast_to`), rotate it with `rope(..., deltas)` and multiply by `rope(ones, 0)`. If test 2 fails for $m>n$, or test 1 for the matrix, check the sign of the sine.
</details>
