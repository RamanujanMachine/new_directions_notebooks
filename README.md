# Recovery notebooks for New Directions in Irrationality

These notebooks accompany the paper

> E. Leibtag, H. Barkan, R. Kalisch, S. Weinbaum, M. Shalyt, I. Kaminer,
> New Directions in Irrationality, [arXiv:2610.27799](https://arxiv.org/abs/2610.27799).

They reproduce the CMF embeddings of classical Apéry-type constructions
(Tables 1–3 and Appendix A of the paper): for each constant, the historical
constructions are realized as trajectories in a conservative matrix field (CMF),
and the finite-depth approximation exponents δ<sub>N</sub> are computed
with exact rational arithmetic.

*These notebooks are experimental.* They measure δ<sub>N</sub> numerically
(up to index 500). Where the paper proves a bound, the measured exponent is
typically better than the proved one, since the proofs use explicit
divisibility bounds that do not capture all cancellation. The proofs of the
paper's theorems are verified separately in the paper's main code repository:
https://github.com/RamanujanMachine/new-directions-in-irrationality

## Contents

- *_history.ipynb: one notebook per constant (ζ(2), ζ(3), ζ(5), ζ(7),
  Catalan's constant, π, logarithms, e, Euler's γ, L(2,χ₋₃), and
  combinations), listing the historical constructions in chronological order.
- shifted_apery.ipynb: the half-shifted Apéry limits (Section 4.3).
- gregory_series.ipynb: the harmonic-weighted Gregory series (Section 7).
- cmfs.py: the special CMFs used by the notebooks that are not plain
  pFq or Meijer G fields.
- *_data.json: cached results for the larger notebooks.

## Running

Python 3.12 with


pip install ramanujantools==0.0.8 sympy==1.14.0 python-flint==0.9.0 matplotlib==3.11.2


Each notebook also has this install line in its first cell. The saved
outputs show the results; some constructions take several minutes at depth 500.
