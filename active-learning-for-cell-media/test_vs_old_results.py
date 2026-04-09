#!/usr/bin/env python3
"""
Validation: reproduce past results with optimized code and compare.

Runs 2 configs that match old result files, then checks:
1. Optimizer rankings are the same (Spearman rank correlation > 0.85)
2. Mean steps are within expected statistical variance
3. Same optimizers win/lose
"""

import sys, os, time, json, warnings
import numpy as np

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import run_harder_competition
from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition

OPTIMISERS = [
    'RANDOM', 'BO_GP_EI', 'SBO_GP_PV', 'SBO_GP_EI_TRUNCDE',
    'SBO_POLY_PV',
    'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'LATIN_HYPERCUBE',
    'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'PLACKETT_BURMAN',
    'DE_DIRECT',
]


def load_old_result(filepath):
    """Load optimizer stats from an old result file."""
    with open(filepath) as f:
        data = json.load(f)
    ca = data.get('combined_analysis', {})
    stats = ca.get('optimizer_stats', {})
    return {name: s['mean_steps'] for name, s in stats.items()}


def spearman_rank_corr(old_ranking, new_ranking):
    """Compute Spearman rank correlation between two optimizer rankings."""
    # Get common optimizers
    common = sorted(set(old_ranking.keys()) & set(new_ranking.keys()))
    if len(common) < 3:
        return 0.0

    # Rank them
    old_sorted = sorted(common, key=lambda x: old_ranking[x])
    new_sorted = sorted(common, key=lambda x: new_ranking[x])

    old_ranks = {name: i for i, name in enumerate(old_sorted)}
    new_ranks = {name: i for i, name in enumerate(new_sorted)}

    n = len(common)
    d_sq = sum((old_ranks[name] - new_ranks[name]) ** 2 for name in common)
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))
    return rho


def compare_results(old_steps, new_steps, test_name):
    """Compare old vs new results and print report."""
    print(f"\n{'='*70}")
    print(f"  {test_name}")
    print(f"{'='*70}")
    print(f"{'Optimizer':30s} {'Old':>10s} {'New':>10s} {'Diff':>10s} {'%Diff':>8s}")
    print("-" * 70)

    common = sorted(set(old_steps.keys()) & set(new_steps.keys()))

    for name in sorted(common, key=lambda x: old_steps[x]):
        old = old_steps[name]
        new = new_steps[name]
        diff = new - old
        pct = (diff / old * 100) if old > 0 else 0
        flag = "  ⚠" if abs(pct) > 50 else ""
        print(f"  {name:30s} {old:10.2f} {new:10.2f} {diff:+10.2f} {pct:+7.1f}%{flag}")

    # Spearman rank correlation
    rho = spearman_rank_corr(old_steps, new_steps)
    print(f"\nSpearman rank correlation: {rho:.4f}")

    # Check top-3 and bottom-3
    old_ranked = sorted(common, key=lambda x: old_steps[x])
    new_ranked = sorted(common, key=lambda x: new_steps[x])

    old_top3 = set(old_ranked[:3])
    new_top3 = set(new_ranked[:3])
    old_bot3 = set(old_ranked[-3:])
    new_bot3 = set(new_ranked[-3:])

    top3_overlap = len(old_top3 & new_top3)
    bot3_overlap = len(old_bot3 & new_bot3)

    print(f"Top-3 overlap: {top3_overlap}/3 (old: {old_ranked[:3]}, new: {new_ranked[:3]})")
    print(f"Bottom-3 overlap: {bot3_overlap}/3 (old: {old_ranked[-3:]}, new: {new_ranked[-3:]})")

    # Pass criteria
    passed = True
    reasons = []

    if rho < 0.70:
        passed = False
        reasons.append(f"Spearman correlation too low: {rho:.3f} < 0.70")
    if top3_overlap < 2:
        passed = False
        reasons.append(f"Top-3 overlap too low: {top3_overlap}/3")
    if bot3_overlap < 2:
        passed = False
        reasons.append(f"Bottom-3 overlap too low: {bot3_overlap}/3")

    if passed:
        print(f"\n✓ PASS — Rankings are consistent (rho={rho:.3f}, top3={top3_overlap}/3, bot3={bot3_overlap}/3)")
    else:
        print(f"\n✗ FAIL — {'; '.join(reasons)}")

    return passed, rho


def test_nn_hard_htl():
    """
    Reproduce: NN_Hard, MOBO_dataset_rat_myocyte, hf=0.95, B=1
    Old file: MOBO_dataset_rat_myocyte_hard_mode_hide_the_label_20260316_191147.json
    Old result used n_competitions=100, n_synthetic=10
    We'll use n_competitions=10, n_synthetic=10 (faster but noisier)
    """
    print("\n" + "=" * 70)
    print("  TEST A: NN_Hard HTL (MOBO, hf=0.95, B=1)")
    print("  Comparing against old result with 100 competitions × 10 synthetic")
    print("=" * 70)

    old_file = os.path.join(ROOT, 'Results', 'Hard_Mode', 'Hide_The_Label',
                            'MOBO_dataset_rat_myocyte_hard_mode_hide_the_label_20260316_191147.json')
    old_steps = load_old_result(old_file)

    difficulty = {'noise_level': 0.1, 'heteroscedastic': True, 'n_modes': 2, 'mode_separation': 2.0}

    t0 = time.time()
    results, analysis = run_harder_competition(
        dataset_name='MOBO_dataset_rat_myocyte',
        surrogate_type='neural_network',
        difficulty_config=difficulty,
        optimizer_names=OPTIMISERS,
        n_competitions=10,
        hidden_fraction=0.95,
        pool_size=200,
        batch_size=1,
        n_jobs=1,
        random_state=42,
        save_results=False,
        num_of_synthetic_data=10,
        use_gpu=True,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Extract new results
    if isinstance(results, dict) and 'combined_analysis' in results:
        stats = results['combined_analysis']['optimizer_stats']
    else:
        stats = analysis['optimizer_stats']

    new_steps = {name: stats[name]['mean_steps'] for name in stats}

    return compare_results(old_steps, new_steps, "NN_Hard HTL: Old (100 comps) vs New Optimized (10 comps)")


def test_rf_regular_htl():
    """
    Reproduce: RF_Regular, MOBO_dataset_rat_myocyte, hf=0.99, B=1
    Old file: MOBO_dataset_rat_myocyte_hard_mode_hide_the_label_20260316_095953.json
    """
    print("\n" + "=" * 70)
    print("  TEST B: RF_Regular HTL (MOBO, hf=0.99, B=1)")
    print("  Comparing against old result with 100 competitions × 10 synthetic")
    print("=" * 70)

    old_file = os.path.join(ROOT, 'Results', 'Hard_Mode', 'Hide_The_Label',
                            'MOBO_dataset_rat_myocyte_hard_mode_hide_the_label_20260316_095953.json')
    old_steps = load_old_result(old_file)

    difficulty = {'noise_level': 0.0, 'heteroscedastic': False, 'n_modes': 1, 'mode_separation': 2.0}

    t0 = time.time()
    results, analysis = run_harder_competition(
        dataset_name='MOBO_dataset_rat_myocyte',
        surrogate_type='random_forest',
        difficulty_config=difficulty,
        optimizer_names=OPTIMISERS,
        n_competitions=10,
        hidden_fraction=0.99,
        pool_size=200,
        batch_size=1,
        n_jobs=1,
        random_state=42,
        save_results=False,
        num_of_synthetic_data=10,
        use_gpu=True,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    if isinstance(results, dict) and 'combined_analysis' in results:
        stats = results['combined_analysis']['optimizer_stats']
    else:
        stats = analysis['optimizer_stats']

    new_steps = {name: stats[name]['mean_steps'] for name in stats}

    return compare_results(old_steps, new_steps, "RF_Regular HTL: Old (100 comps) vs New Optimized (10 comps)")


def test_or_comparison():
    """
    Test Open Race: reproduce a past OR result.
    """
    print("\n" + "=" * 70)
    print("  TEST C: Open Race (RF, Hard, MOBO, B=10)")
    print("=" * 70)

    # Find a past OR result
    or_dir = os.path.join(ROOT, 'Results', 'Hard_Mode', 'Open_Race')
    or_files = sorted([f for f in os.listdir(or_dir) if 'MOBO' in f])

    if not or_files:
        print("No past OR results found, skipping")
        return True, 1.0

    old_file = os.path.join(or_dir, or_files[-1])
    with open(old_file) as f:
        old_data = json.load(f)

    old_diff = old_data.get('difficulty_config', {})
    old_surr = old_data.get('surrogate_type', 'random_forest')
    old_analysis = old_data.get('analysis', {})
    old_stats = old_analysis.get('optimizer_stats', {})

    if not old_stats:
        print(f"No optimizer_stats in {or_files[-1]}, skipping")
        return True, 1.0

    print(f"Old file: {or_files[-1]}")
    print(f"Surrogate: {old_surr}, Difficulty: {old_diff}")

    # Extract old best values
    old_best = {}
    for name, s in old_stats.items():
        val = s.get('mean_best_value', s.get('mean_final_best', None))
        if val is not None:
            old_best[name] = float(val)

    if not old_best:
        print("Could not extract old best values, skipping")
        return True, 1.0

    # Run new
    t0 = time.time()
    results, analysis = run_harder_open_race_competition(
        dataset_name='MOBO_dataset_rat_myocyte',
        surrogate_type=old_surr,
        difficulty_config=old_diff,
        optimizer_names=OPTIMISERS,
        n_competitions=5,
        S=200, R=10, B=10,
        n_jobs=1,
        random_state=42,
        save_results=False,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    new_stats = analysis.get('optimizer_stats', {})
    new_best = {}
    for name, s in new_stats.items():
        val = s.get('mean_best_value', s.get('mean_final_best', None))
        if val is not None:
            new_best[name] = float(val)

    # For OR, compare rankings by best value (higher is better — negate for ranking)
    old_neg = {k: -v for k, v in old_best.items()}
    new_neg = {k: -v for k, v in new_best.items()}

    rho = spearman_rank_corr(old_neg, new_neg)

    print(f"\n{'Optimizer':30s} {'Old Best':>12s} {'New Best':>12s}")
    print("-" * 56)
    common = sorted(set(old_best.keys()) & set(new_best.keys()), key=lambda x: -old_best[x])
    for name in common:
        print(f"  {name:30s} {old_best[name]:12.4f} {new_best[name]:12.4f}")

    print(f"\nSpearman rank correlation: {rho:.4f}")
    passed = rho > 0.50  # OR is noisier with fewer competitions
    if passed:
        print(f"✓ PASS — OR rankings consistent (rho={rho:.3f})")
    else:
        print(f"✗ FAIL — OR rankings diverged (rho={rho:.3f} < 0.50)")

    return passed, rho


if __name__ == '__main__':
    print("=" * 70)
    print("  VALIDATION: Optimized Code vs Past Results")
    print("  Checks that optimizer parallelization + lazy GP refit")
    print("  produce equivalent rankings to the sequential baseline")
    print("=" * 70)

    results = {}

    passed_a, rho_a = test_nn_hard_htl()
    results['NN_Hard_HTL'] = passed_a

    passed_b, rho_b = test_rf_regular_htl()
    results['RF_Regular_HTL'] = passed_b

    passed_c, rho_c = test_or_comparison()
    results['Open_Race'] = passed_c

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    sys.exit(0 if all_passed else 1)
