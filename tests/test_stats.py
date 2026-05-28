"""Tests for the lightweight stats helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.stats import (
    mean_ci95,
    paired_t_pvalue,
    wilcoxon_signed_rank_pvalue,
)


def test_mean_ci95_zero_when_constant():
    mu, half = mean_ci95([1.0, 1.0, 1.0])
    assert mu == 1.0
    assert half == 0.0


def test_paired_t_detects_consistent_difference():
    x = [1.1, 1.2, 1.3, 1.4, 1.5]
    y = [0.1, 0.2, 0.3, 0.4, 0.5]
    p = paired_t_pvalue(x, y)
    assert p < 0.01


def test_wilcoxon_detects_consistent_difference():
    x = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    p = wilcoxon_signed_rank_pvalue(x, y)
    assert p < 0.1


def test_paired_t_handles_no_difference():
    x = [1.0, 1.0, 1.0]
    y = [1.0, 1.0, 1.0]
    p = paired_t_pvalue(x, y)
    assert p == 1.0


if __name__ == "__main__":
    test_mean_ci95_zero_when_constant()
    test_paired_t_detects_consistent_difference()
    test_wilcoxon_detects_consistent_difference()
    test_paired_t_handles_no_difference()
    print("stats tests passed")
