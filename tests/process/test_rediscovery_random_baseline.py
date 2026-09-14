import math
from itertools import combinations
import pytest

from src.process.benchmarks.rediscovery import (
    calculate_hypergeometric_baseline,
    calculate_conditional_hypergeometric_baseline,
)


def test_top1_formula_exact():
    # N=26 candidates, N_init=3, budget=5
    # N_rem = 23
    curve = calculate_hypergeometric_baseline(total_candidates=26, initial_size=3, budget=5, top_k=1)
    for s in range(1, 6):
        expected = s / 23.0
        assert pytest.approx(curve[s], 1e-9) == expected


def test_top_k_unconditional_matches_brute_force_enumeration():
    N = 6
    N_init = 2
    budget = 3
    k = 2

    pool = list(range(N))
    top_k_set = {0, 1}
    init_eligible = [c for c in pool if c != 0]  # 0 is top-1, excluded from init
    all_inits = list(combinations(init_eligible, N_init))

    # Calculate via formula
    formula_curve = calculate_hypergeometric_baseline(total_candidates=N, initial_size=N_init, budget=budget, top_k=k)

    # Brute force enumeration over all possible inits and sequential draws
    for s in range(1, budget + 1):
        bf_hits = 0
        bf_total = 0
        for init in all_inits:
            rem_pool = [c for c in pool if c not in init]
            for draw in combinations(rem_pool, s):
                bf_total += 1
                if any(d in top_k_set for d in draw):
                    bf_hits += 1
        bf_prob = bf_hits / bf_total
        assert pytest.approx(formula_curve[s], 1e-9) == bf_prob


def test_top_k_conditional_matches_brute_force_enumeration():
    N = 6
    N_init = 2
    budget = 3
    top_k_set = {0, 1}
    pool = list(range(N))
    init_eligible = [c for c in pool if c != 0]
    all_inits = list(combinations(init_eligible, N_init))

    # For each init design, determine number of remaining targets
    rem_targets = [len([c for c in top_k_set if c not in init]) for init in all_inits]

    cond_curve = calculate_conditional_hypergeometric_baseline(
        total_candidates=N,
        initial_designs_remaining_targets=rem_targets,
        budget=budget,
        initial_size=N_init,
    )

    for s in range(1, budget + 1):
        bf_hits = 0
        bf_total = 0
        for init in all_inits:
            rem_pool = [c for c in pool if c not in init]
            for draw in combinations(rem_pool, s):
                bf_total += 1
                if any(d in top_k_set for d in draw):
                    bf_hits += 1
        bf_prob = bf_hits / bf_total
        assert pytest.approx(cond_curve[s], 1e-9) == bf_prob


def test_edge_cases():
    # Budget exceeds remaining pool
    c = calculate_hypergeometric_baseline(total_candidates=5, initial_size=3, budget=5, top_k=1)
    assert c[1] == pytest.approx(1 / 2.0)
    assert c[2] == pytest.approx(1.0)
    assert c[3] == pytest.approx(1.0)

    # Empty / zero candidates
    assert calculate_hypergeometric_baseline(0, 0, 5) == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
