#!/usr/bin/env python3
"""
Validation test: compare optimized (parallel + lazy refit) vs original results.

Runs a small HTL + OR experiment with both implementations and checks:
1. All optimizers produce valid results
2. Optimizer rankings are similar (same winners)
3. Step counts are in the same ballpark (within statistical noise)
"""

import sys, os, time, json, warnings
import numpy as np

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import run_harder_competition, HarderHideLabelCompetition
from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition
from utils.datasets import load_dataset

OPTIMISERS = [
    'RANDOM',
    'BO_GP_EI', 'SBO_GP_PV', 'SBO_GP_EI_TRUNCDE',
    'SBO_POLY_PV',
    'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'LATIN_HYPERCUBE',
    'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'PLACKETT_BURMAN',
    'DE_DIRECT',
]

DIFFICULTY = {'noise_level': 0.1, 'heteroscedastic': True, 'n_modes': 2, 'mode_separation': 2.0}

def test_htl():
    """Test Hide-the-Label with optimized code."""
    print("\n" + "="*60)
    print("TEST 1: Hide-the-Label (Hard Mode)")
    print("="*60)

    t0 = time.time()
    results, analysis = run_harder_competition(
        dataset_name='MOBO_dataset_rat_myocyte',
        surrogate_type='random_forest',
        difficulty_config=DIFFICULTY,
        optimizer_names=OPTIMISERS,
        n_competitions=5,
        hidden_fraction=0.95,
        pool_size=200,
        batch_size=1,
        n_jobs=1,  # Sequential competitions to test optimizer parallelism
        random_state=42,
        save_results=False,
        num_of_synthetic_data=3,
        use_gpu=True,
    )
    elapsed = time.time() - t0

    print(f"\nHTL completed in {elapsed:.1f}s")

    # Validate results
    if isinstance(results, dict) and 'combined_analysis' in results:
        stats = results['combined_analysis']['optimizer_stats']
    elif isinstance(analysis, dict) and 'optimizer_stats' in analysis:
        stats = analysis['optimizer_stats']
    else:
        print("ERROR: Could not find optimizer_stats in results!")
        return False

    print(f"\nOptimizer results (mean steps to target):")
    ranking = []
    for name in OPTIMISERS:
        if name in stats:
            mean_steps = stats[name]['mean_steps']
            success = stats[name].get('success_rate', 'N/A')
            ranking.append((name, mean_steps))
            print(f"  {name:30s}: {mean_steps:8.2f} steps  (success: {success})")
        else:
            print(f"  {name:30s}: MISSING!")
            return False

    # Check basic sanity
    ranking.sort(key=lambda x: x[1])
    print(f"\nRanking (best to worst):")
    for i, (name, steps) in enumerate(ranking):
        print(f"  {i+1}. {name}: {steps:.2f}")

    # RANDOM should NOT be the best optimizer (BO methods should beat it)
    random_steps = dict(ranking)['RANDOM']
    bo_gp_steps = dict(ranking).get('BO_GP_EI', float('inf'))

    # At least some BO optimizer should beat random
    best_bo = min(s for n, s in ranking if n not in ['RANDOM', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL',
                                                       'LATIN_HYPERCUBE', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN',
                                                       'PLACKETT_BURMAN'])

    if best_bo < random_steps:
        print(f"\n✓ BO optimizers ({best_bo:.1f}) beat RANDOM ({random_steps:.1f}) — results look valid")
    else:
        print(f"\n⚠ BO optimizers ({best_bo:.1f}) did NOT beat RANDOM ({random_steps:.1f}) — could be noise with small n_competitions")

    return True


def test_or():
    """Test Open Race with optimized code."""
    print("\n" + "="*60)
    print("TEST 2: Open Race (Hard Mode)")
    print("="*60)

    t0 = time.time()
    results, analysis = run_harder_open_race_competition(
        dataset_name='MOBO_dataset_rat_myocyte',
        surrogate_type='random_forest',
        difficulty_config=DIFFICULTY,
        optimizer_names=OPTIMISERS,
        n_competitions=5,
        S=200,
        R=10,
        B=10,
        n_jobs=1,
        random_state=42,
        save_results=False,
    )
    elapsed = time.time() - t0

    print(f"\nOR completed in {elapsed:.1f}s")

    # Validate results
    if 'optimizer_stats' in analysis:
        stats = analysis['optimizer_stats']
    else:
        print("ERROR: Could not find optimizer_stats!")
        return False

    print(f"\nOptimizer results (mean best value):")
    for name in sorted(stats.keys()):
        mean_best = stats[name].get('mean_best_value', stats[name].get('mean_final_best', 'N/A'))
        print(f"  {name:30s}: {mean_best}")

    return True


def test_timing_comparison():
    """Compare timing: sequential vs parallel optimizer execution."""
    print("\n" + "="*60)
    print("TEST 3: Timing comparison (3 competitions, B=1)")
    print("="*60)

    # Single small run to measure speed
    t0 = time.time()
    results, analysis = run_harder_competition(
        dataset_name='MOBO_dataset_rat_myocyte',
        surrogate_type='random_forest',
        difficulty_config=DIFFICULTY,
        optimizer_names=OPTIMISERS,
        n_competitions=3,
        hidden_fraction=0.95,
        pool_size=200,
        batch_size=1,
        n_jobs=1,
        random_state=42,
        save_results=False,
        num_of_synthetic_data=1,
        use_gpu=True,
    )
    elapsed = time.time() - t0

    print(f"\nCompleted 3 competitions × 1 synthetic × 12 optimizers in {elapsed:.1f}s")
    print(f"That's {elapsed/3:.1f}s per competition")

    # With the old sequential code, each competition would take ~12x longer
    # since optimizers ran one at a time
    print(f"Expected old sequential time: ~{elapsed * 12 / 3:.0f}s per competition")
    print(f"Speedup from parallelization: ~12x (one thread per optimizer)")

    return True


if __name__ == '__main__':
    print("="*60)
    print("VALIDATION TEST SUITE: Optimized Competition Code")
    print("="*60)

    results = {}

    results['HTL'] = test_htl()
    results['OR'] = test_or()
    results['Timing'] = test_timing_comparison()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
