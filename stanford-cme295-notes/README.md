# Stanford CME 295 notes

Hands-on numpy exercises written while working through Stanford CME 295 "Transformers & Large Language Models" (Autumn 2025, A. Amidi & S. Amidi), one exercise per lecture. Each exercise has a skeleton to fill in, pytest tests that fail until you implement it, and a hidden reference solution.

## How the exercises are produced

An agent pipeline: local Whisper transcription of the lecture, a study note, an exercise, and then a fact-check agent that verifies every claim against the lecture before anything is used. Tests are labelled by how well supported they are: `lecture_claim_` (stated in the lecture), `lecture_consequence_` (follows from it), `own_addition_` (beyond the lecture) and `technical_` (implementation constraints). The pipeline code is not in this repository yet.

## Exercises

| Lecture | Topic | Exercise dir | What you implement | Tests |
|---|---|---|---|---|
| 1 | Transformer and attention | [`exercises/01-transformer`](exercises/01-transformer/) | softmax, scaled dot-product and multi-head attention, causal mask, sinusoidal positions | 25 |
| 2 | Models and tricks (positional encodings) | [`exercises/02-models-and-tricks`](exercises/02-models-and-tricks/) | sinusoidal embeddings, explicit rotation matrix, RoPE | 24 |

Run an exercise from this directory, one at a time:

```
uv run --with numpy --with pytest pytest exercises/01-transformer
```

or `pip install numpy pytest && pytest exercises/01-transformer`. Before you implement anything, the tests fail with `NotImplementedError`.

## What is deliberately not here

Lecture transcripts, audio, slides and my private study notes. Please use the official sources:

- Course page: https://cme295.stanford.edu
- Lecture 1: https://www.youtube.com/watch?v=Ub3GoFaUcds
- Lecture 2: https://www.youtube.com/watch?v=yT84Y5zCnaA
- The authors' official cheatsheet: https://github.com/afshinea/stanford-cme-295-transformers-large-language-models

Notes are my own; the recordings and slides belong to the course authors.
