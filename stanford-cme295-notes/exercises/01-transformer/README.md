# Exercise: attention in pure numpy

Time: 30-60 minutes. You need Python and numpy; no prior knowledge of transformers is required, everything you need is below.

## Goal

Implement scaled dot-product attention and multi-head attention, then check four things related to the lecture with experiments:

1. the attention weights are a probability distribution (every row sums to 1),
2. the causal mask cuts off the future,
3. without positional encoding, attention is blind to token order (a consequence of the lecture at 1:19:42, not a claim it makes directly),
4. dividing by $\sqrt{d_k}$ keeps the scale of the dot products constant (a consequence of the claim at 1:33:51; the lecture says only that the dot products grow with the dimension, and the exact $\sqrt{d_k}$ scale is the author's own calculation).

In addition, as an own addition beyond the lecture, you observe what not scaling does to the softmax.

What we left out: Add & Norm layers, the feed-forward network, training and the whole decoder. Only attention itself is covered. The lecture only mentions positional encoding, so you get the formula below.

## Matrix shapes

One token is a vector of numbers (an embedding). A sequence of $n$ tokens is a matrix in which row = token.

| Symbol | Shape | Meaning |
|---|---|---|
| $X$ | $(n, d_{model})$ | embeddings; here $n=8$, $d_{model}=16$ |
| $W_Q, W_K, W_V$ (one head) | $(d_{model}, d_k)$ | projections; here $d_k=4$ |
| $Q = XW_Q$, $K = XW_K$, $V = XW_V$ | $(n, d_k)$ | queries, keys, values: row $i$ is the role of token $i$ |
| $QK^T$ | $(n, n)$ | element $[i,j]$ = dot product of query $i$ with key $j$ = "how well $i$ matches $j$" |
| weights $=\mathrm{softmax}(QK^T/\sqrt{d_k})$ | $(n, n)$ | softmax applied to each row separately |
| output $=$ weights $\cdot V$ | $(n, d_k)$ | token $i$ = weighted average of the values of all tokens |

Formula: $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\dfrac{QK^T}{\sqrt{d_k}}\right)V$.

Multi-head attention: $h=4$ heads, each with its own $W_Q,W_K,W_V$ (in the code, arrays of shape $(h, d_{model}, d_k)$; head $i$ is `W_Q[i]`). Each head returns $(n, d_k)$. Concatenate them along the feature axis to $(n, h\cdot d_k)=(8,16)$ and multiply by $W_O$ of shape $(h\cdot d_k, d_{model})$. The result is again $(n, d_{model})$.

Causal mask: token $i$ must not look at $j>i$. Before the softmax, put $-\infty$ in the elements above the diagonal of the score matrix. After the softmax they get weight 0.

Positional encoding (sinusoidal), $PE$ of shape $(n, d_{model})$:
$PE_{pos,2i}=\sin(pos/10000^{2i/d_{model}})$, $PE_{pos,2i+1}=\cos(pos/10000^{2i/d_{model}})$. You add it to $X$.

About this encoding the lecture says only "sines and cosines" (1:29:34). The formula comes from the original Transformer paper and is not discussed in the lecture.

## What to build

File `task.py`: fill in the functions `softmax`, `attention`, `multi_head`, `sinusoidal_positions`, `score_std`, `mean_max_weight`. `make_inputs` and `report` are ready (`report` calls your functions).

- Input: data from `make_inputs()` (seed 0), the `causal` flag, the `scale` flag.
- Output: `attention` returns `(output (n, d_k), weights (n, n))`, `multi_head` returns `(n, d_model)`.
- Report: `python task.py` prints the weights of one head, the standard deviation of the $QK^T$ dot products and (own addition) the mean maximum softmax weight, each for $d_k=4$ and $d_k=512$, with and without dividing by $\sqrt{d_k}$.
- Distribution in the experiments: the components of $Q$ and $K$ are independent, drawn from $N(0,1)$ (`rng.normal`), $n=16$, 200 trials.

## Passing criteria

Four substantive criteria (1-4) plus one technical criterion (5). All tests pass:

1. the weight rows sum to 1 (tolerance $10^{-6}$), `multi_head` returns $(n, d_{model})$,
2. with `causal=True` the weights above the diagonal are 0, and changing the last token does not change the outputs of earlier positions,
3. consequence of the lecture (1:19:42): with `causal=False` and no positional encoding, permuting the rows of $X$ permutes the output in the same way; after adding the encoding this equality disappears,
4. consequence of the claim at 1:33:51: for $Q, K$ with i.i.d. $N(0,1)$ components, the standard deviation of the raw dot products grows with $d_k$ (compare 4 and 512), and after dividing by $\sqrt{d_k}$ it is approximately constant. The lecture says only that the dot products grow with the dimension; the exact values (about $\sqrt{d_k}$ without dividing, about 1 after dividing) are the author's own calculation,
5. technical: `attention` and `softmax` contain no loops or comprehensions, only matrix operations.

Own addition, beyond the lecture (also tested): without dividing, the mean maximum softmax weight for $d_k=512$ is clearly closer to 1 than with dividing.

## How to run

From the repository directory `stanford-cme295-notes/`, run each exercise separately (both test files are named `test_task.py`):

```
uv run --with numpy --with pytest pytest exercises/01-transformer
uv run --with numpy python exercises/01-transformer/task.py
```

or, without uv:

```
pip install numpy pytest && pytest exercises/01-transformer
```

The reference solution is hidden in `solution/solution.py`. To check it, copy it over `task.py` (or into a temporary directory next to `test_task.py` as `task.py`).

## Hints

Read them one at a time, only when you are stuck.

<details>
<summary>Hint 1: softmax and the mask</summary>

Softmax: subtract the row maximum (`keepdims=True`), exponentiate, divide by the sum over `axis=-1`. Build the mask from `np.triu(np.ones((n, n), dtype=bool), k=1)` and insert `-np.inf` with `np.where`. The diagonal stays, so every row has something finite.
</details>

<details>
<summary>Hint 2: order of operations in attention and multi_head</summary>

`scores = Q @ K.T`, then division by `np.sqrt(d_k)` (when `scale`), then the mask, then the softmax, and finally `weights @ V`. In `multi_head` make a list of the head outputs, `np.concatenate(..., axis=-1)`, and `@ W_O`.
</details>

<details>
<summary>Hint 3: positional encoding and experiment 4</summary>

Positions: `pos = np.arange(n)[:, None]`, `two_i = np.arange(0, d_model, 2)[None, :]`, angles = `pos / 10000 ** (two_i / d_model)`; put the sine into columns `0::2` and the cosine into `1::2`. In `score_std` draw `Q`, `K` with `rng.normal(size=(n, d_k))`, compute `Q @ K.T` (optionally divide by `np.sqrt(d_k)`), collect the trials and take `np.std`. For the own addition (`mean_max_weight`) call `attention(Q, K, K, scale=scale)` and average `w.max(axis=1)`; without scaling the dot products have a standard deviation of about $\sqrt{d_k}$, so the softmax is almost one-hot (own addition, beyond the lecture).
</details>
