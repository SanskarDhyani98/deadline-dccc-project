"""Lightweight statistics for paired multi-seed comparison.

Pure-Python implementations so the project doesn't pick up a hard
dependency on scipy. Sample sizes are small (5-30 seeds) so analytic
t/Wilcoxon implementations are accurate enough; we report both.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def mean_ci95(xs: Iterable[float]) -> Tuple[float, float]:
    """Return (mean, half-width of 95% CI) using the normal approximation.

    For n < 30 a t-multiplier would be slightly larger; the small
    overestimate of significance is conservative for our use here
    (rejecting null is *harder* with a larger CI).
    """
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    mu = _mean(xs)
    sd = _stdev(xs)
    se = sd / math.sqrt(len(xs))
    return mu, 1.96 * se


def paired_t_pvalue(x: List[float], y: List[float]) -> float:
    """Two-sided paired t-test p-value via normal approximation."""
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    if n < 2:
        return 1.0
    mu = _mean(diffs)
    sd = _stdev(diffs)
    if sd == 0:
        return 0.0 if mu != 0 else 1.0
    t = mu / (sd / math.sqrt(n))
    # Two-sided p-value from standard normal CDF approximation.
    z = abs(t)
    # Abramowitz & Stegun 7.1.26 approximation for erf.
    return 2.0 * (1.0 - _phi(z))


def wilcoxon_signed_rank_pvalue(x: List[float], y: List[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value (normal approximation)."""
    diffs = [a - b for a, b in zip(x, y) if a - b != 0]
    n = len(diffs)
    if n < 1:
        return 1.0
    abs_diffs = sorted(((abs(d), 1 if d > 0 else -1) for d in diffs), key=lambda t: t[0])
    # Average ranks for ties.
    ranks: List[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    w_plus = sum(r for r, (_, sign) in zip(ranks, abs_diffs) if sign > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0:
        return 1.0
    z = (w_plus - mean_w) / math.sqrt(var_w)
    return 2.0 * (1.0 - _phi(abs(z)))


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
