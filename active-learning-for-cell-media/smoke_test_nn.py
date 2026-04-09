"""
Smoke test: Hard Mode with Neural Network surrogate on one dataset.
Checks whether BO methods still outperform DOE/heuristics.
"""
import sys
import os
import logging
logging.disable(logging.CRITICAL)  # suppress all log output
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import run_harder_competition
from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition

DATASET = 'DBO_dataset_rat_myocyte'
SURROGATE = 'neural_network'
# NN surrogate does not accept n_discontinuities — keep config compatible
DIFFICULTY = {'noise_level': 0.1, 'heteroscedastic': True, 'n_modes': 2}
OPTIMISERS = [
    'RANDOM',
    'BO_GP_EI', 'SBO_GP_PV', 'SBO_GP_EI_TRUNCDE',
    'SBO_ANN_PV', 'SMART_BO',
    'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'LATIN_HYPERCUBE',
    'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'PLACKETT_BURMAN',
    'DE_DIRECT',
]

print("=" * 60)
print(f"Smoke test: Hard NN surrogate on {DATASET}")
print("=" * 60)

# --- Hide-the-Label ---
print("\n[1/2] Running Hide-the-Label (batch=10, 5 synthetic datasets, 5 competitions)...")
htl_results, htl_analysis = run_harder_competition(
    dataset_name=DATASET,
    surrogate_type=SURROGATE,
    difficulty_config=DIFFICULTY,
    optimizer_names=OPTIMISERS,
    n_competitions=5,
    hidden_fraction=0.95,
    pool_size=200,
    batch_size=10,
    n_jobs=-1,
    random_state=42,
    save_results=True,
    num_of_synthetic_data=5,
    use_gpu=False,
)

print("\n--- Hide-the-Label mean steps (lower = better) ---")
if 'optimizer_stats' in htl_results:
    stats = htl_results['optimizer_stats']
    ranked = sorted(stats.items(), key=lambda x: x[1].get('mean', 9999))
    for rank, (opt, s) in enumerate(ranked, 1):
        print(f"  {rank:2d}. {opt:<30s}  mean={s.get('mean', float('nan')):.2f}  median={s.get('median', float('nan')):.1f}")
else:
    print("  (raw results saved — check results folder)")

# --- Open Race ---
print("\n[2/2] Running Open Race (B=10, 5 competitions)...")
or_results, or_analysis = run_harder_open_race_competition(
    dataset_name=DATASET,
    surrogate_type=SURROGATE,
    difficulty_config=DIFFICULTY,
    optimizer_names=OPTIMISERS,
    n_competitions=5,
    S=200,
    R=10,
    B=10,
    n_jobs=-1,
    random_state=42,
    save_results=True,
)

print("\n--- Open Race best value found (higher = better) ---")
if 'optimizer_stats' in or_results:
    stats = or_results['optimizer_stats']
    ranked = sorted(stats.items(), key=lambda x: -x[1].get('mean', -9999))
    for rank, (opt, s) in enumerate(ranked, 1):
        print(f"  {rank:2d}. {opt:<30s}  mean={s.get('mean', float('nan')):.4f}  median={s.get('median', float('nan')):.4f}")
else:
    print("  (raw results saved — check results folder)")

print("\n=== Smoke test complete ===")
