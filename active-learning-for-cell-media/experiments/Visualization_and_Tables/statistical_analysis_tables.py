#!/usr/bin/env python3
"""
Statistical Analysis Tables Generator

This script generates comprehensive statistical analysis tables for the AL Project
experimental results, following the statistical framework described in the paper.

For each of the 4 settings (Hard_Mode/Hide_The_Label, Hard_Mode/Open_Race, 
Regular_Mode/Hide_The_Label, Regular_Mode/Open_Race), it computes:
- Friedman test results
- Pairwise permutation tests
- Hodges-Lehmann median paired differences
- Bootstrap confidence intervals
- Dataset-level aggregations
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from itertools import combinations
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Performs statistical analysis on optimizer competition results."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.results = {}
        
    def load_json_files(self, folder_path: str) -> Dict[str, Any]:
        """Load all JSON files from a folder."""
        files_data = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    files_data[filename] = json.load(f)
        return files_data
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse the structured filename to extract metadata."""
        parts = filename.replace('.json', '').split('_')
        
        # Find the batch number
        batch = None
        for i, part in enumerate(parts):
            if part.startswith('Batch'):
                batch = part
                break
        
        # Find dataset name (everything before mode type)
        dataset_end_idx = None
        for i, part in enumerate(parts):
            if part in ['Hard', 'Easy']:
                dataset_end_idx = i
                break
        
        if dataset_end_idx:
            dataset = '_'.join(parts[:dataset_end_idx])
        else:
            dataset = 'Unknown'
        
        return {
            'dataset': dataset,
            'batch': batch if batch else 'Unknown',
            'filename': filename
        }
    
    def extract_hide_the_label_data(self, files_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract steps_to_target data from Hide_The_Label experiments."""
        records = []
        
        for filename, data in files_data.items():
            metadata = self.parse_filename(filename)
            dataset = metadata['dataset']
            batch = metadata['batch']
            
            # Handle both 'tournament_results' and 'all_tournament_results' keys
            if 'tournament_results' in data:
                tournament_results = data['tournament_results']
            elif 'all_tournament_results' in data:
                tournament_results = data['all_tournament_results']
            else:
                continue
            
            # Handle both dict and list structures
            if isinstance(tournament_results, list):
                for tournament in tournament_results:
                    if 'competitions' not in tournament:
                        continue
                        
                    for comp_idx, competition in enumerate(tournament['competitions']):
                        if 'optimizer_results' not in competition:
                            continue
                            
                        for optimizer, result in competition['optimizer_results'].items():
                            steps = result.get('steps_to_target', np.nan)
                            records.append({
                                'dataset': dataset,
                                'batch': batch,
                                'competition_idx': comp_idx,
                                'optimizer': optimizer,
                                'steps_to_target': steps,
                                'instance': f"{dataset}_{batch}_comp{comp_idx}"
                            })
            elif isinstance(tournament_results, dict):
                # Handle case where tournament_results is a dict with competitions
                if 'competitions' in tournament_results:
                    for comp_idx, competition in enumerate(tournament_results['competitions']):
                        if 'optimizer_results' not in competition:
                            continue
                            
                        for optimizer, result in competition['optimizer_results'].items():
                            steps = result.get('steps_to_target', np.nan)
                            records.append({
                                'dataset': dataset,
                                'batch': batch,
                                'competition_idx': comp_idx,
                                'optimizer': optimizer,
                                'steps_to_target': steps,
                                'instance': f"{dataset}_{batch}_comp{comp_idx}"
                            })
        
        return pd.DataFrame(records)
    
    def extract_open_race_data(self, files_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract final_best data from Open_Race experiments."""
        records = []
        
        for filename, data in files_data.items():
            metadata = self.parse_filename(filename)
            dataset = metadata['dataset']
            batch = metadata['batch']
            
            if 'tournament_results' not in data:
                continue
            
            tournament_results = data['tournament_results']
            
            # Handle both dict and list structures
            if isinstance(tournament_results, dict):
                # For Open Race, tournament_results is a dict with 'competitions' key
                if 'competitions' in tournament_results:
                    for comp_idx, competition in enumerate(tournament_results['competitions']):
                        if 'optimizer_results' not in competition:
                            continue
                            
                        for optimizer, result in competition['optimizer_results'].items():
                            # Get the best value from optimization history
                            best_value = result.get('best_value_so_far', np.nan)
                            if isinstance(best_value, list) and len(best_value) > 0:
                                best_value = best_value[-1]  # Take final value
                            elif 'optimization_history' in result and len(result['optimization_history']) > 0:
                                best_value = result['optimization_history'][-1].get('best_value_so_far', np.nan)
                            
                            records.append({
                                'dataset': dataset,
                                'batch': batch,
                                'competition_idx': comp_idx,
                                'optimizer': optimizer,
                                'final_best': best_value,
                                'instance': f"{dataset}_{batch}_comp{comp_idx}"
                            })
            elif isinstance(tournament_results, list):
                # For Hide_The_Label, tournament_results is a list
                for tournament in tournament_results:
                    if 'competitions' not in tournament:
                        continue
                        
                    for comp_idx, competition in enumerate(tournament['competitions']):
                        if 'optimizer_results' not in competition:
                            continue
                            
                        for optimizer, result in competition['optimizer_results'].items():
                            # Get the best value from optimization history
                            best_value = result.get('best_value_so_far', np.nan)
                            if isinstance(best_value, list) and len(best_value) > 0:
                                best_value = best_value[-1]
                            elif 'optimization_history' in result and len(result['optimization_history']) > 0:
                                best_value = result['optimization_history'][-1].get('best_value_so_far', np.nan)
                            
                            records.append({
                                'dataset': dataset,
                                'batch': batch,
                                'competition_idx': comp_idx,
                                'optimizer': optimizer,
                                'final_best': best_value,
                                'instance': f"{dataset}_{batch}_comp{comp_idx}"
                            })
        
        return pd.DataFrame(records)
    
    def friedman_test(self, df: pd.DataFrame, metric: str) -> Tuple[float, float]:
        """Perform Friedman test on the data."""
        # Pivot to get instances x optimizers
        pivot = df.pivot_table(index='instance', columns='optimizer', values=metric)
        
        # Remove any rows with NaN values
        pivot = pivot.dropna()
        
        if len(pivot) < 2:
            return np.nan, np.nan
        
        # Perform Friedman test
        optimizer_data = [pivot[col].values for col in pivot.columns]
        
        try:
            statistic, p_value = friedmanchisquare(*optimizer_data)
            return statistic, p_value
        except:
            return np.nan, np.nan
    
    def permutation_test(self, diff_values: np.ndarray, n_permutations: int = 1000) -> float:
        """Perform two-sided sign-flip permutation test."""
        if len(diff_values) == 0 or np.all(np.isnan(diff_values)):
            return np.nan
        
        # Remove NaN values
        diff_values = diff_values[~np.isnan(diff_values)]
        
        if len(diff_values) == 0:
            return np.nan
        
        # Observed test statistic (mean of differences)
        observed_stat = np.mean(diff_values)
        
        # Vectorized permutation test (much faster)
        np.random.seed(42)  # For reproducibility
        signs = np.random.choice([-1, 1], size=(n_permutations, len(diff_values)))
        null_stats = np.mean(diff_values * signs, axis=1)
        
        # Two-sided p-value
        p_value = np.mean(np.abs(null_stats) >= np.abs(observed_stat))
        return p_value
    
    def hodges_lehmann_estimator(self, diff_values: np.ndarray) -> float:
        """Calculate Hodges-Lehmann median paired difference."""
        diff_values = diff_values[~np.isnan(diff_values)]
        if len(diff_values) == 0:
            return np.nan
        return np.median(diff_values)
    
    def bootstrap_ci(self, diff_values: np.ndarray, n_bootstrap: int = 1000, 
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        diff_values = diff_values[~np.isnan(diff_values)]
        if len(diff_values) == 0:
            return np.nan, np.nan
        
        # Vectorized bootstrap (much faster)
        np.random.seed(42)  # For reproducibility
        n = len(diff_values)
        bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
        bootstrap_samples = diff_values[bootstrap_indices]
        bootstrap_medians = np.median(bootstrap_samples, axis=1)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
        upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))
        return lower, upper
    
    def pairwise_comparisons(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Perform pairwise comparisons between all optimizer pairs."""
        optimizers = df['optimizer'].unique()
        results = []
        
        total_pairs = len(list(combinations(optimizers, 2)))
        print(f"  Computing {total_pairs} pairwise comparisons...")
        
        for idx, (opt_a, opt_b) in enumerate(combinations(optimizers, 2)):
            if idx % 10 == 0:
                print(f"    Progress: {idx}/{total_pairs}", end='\r')
            
            # Get data for both optimizers
            df_a = df[df['optimizer'] == opt_a].set_index('instance')[metric]
            df_b = df[df['optimizer'] == opt_b].set_index('instance')[metric]
            
            # Align on common instances
            common_instances = df_a.index.intersection(df_b.index)
            if len(common_instances) == 0:
                continue
            
            diff_values = (df_a.loc[common_instances] - df_b.loc[common_instances]).values
            
            # Calculate statistics
            hl_median = self.hodges_lehmann_estimator(diff_values)
            p_value = self.permutation_test(diff_values)
            ci_lower, ci_upper = self.bootstrap_ci(diff_values)
            
            results.append({
                'Optimizer_A': opt_a,
                'Optimizer_B': opt_b,
                'HL_Median': hl_median,
                'P_Value': p_value,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'N_Instances': len(common_instances)
            })
        
        print(f"    Progress: {total_pairs}/{total_pairs} done")
        return pd.DataFrame(results)
    
    def holm_bonferroni_correction(self, pairwise_df: pd.DataFrame) -> pd.DataFrame:
        """Apply Holm-Bonferroni correction for multiple comparisons."""
        df = pairwise_df.copy()
        df = df.sort_values('P_Value')
        
        n = len(df)
        adjusted_pvals = []
        
        for i, p in enumerate(df['P_Value']):
            adjusted_p = min(p * (n - i), 1.0)
            adjusted_pvals.append(adjusted_p)
        
        df['Adjusted_P_Value'] = adjusted_pvals
        return df
    
    def dataset_level_analysis(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Aggregate results at dataset level and perform analysis."""
        # Group by dataset and optimizer, take median across competitions
        agg_df = df.groupby(['dataset', 'optimizer'])[metric].median().reset_index()
        agg_df['instance'] = agg_df['dataset']
        
        return agg_df
    
    def optimizer_summary_statistics(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Calculate summary statistics for each optimizer."""
        summary = df.groupby('optimizer')[metric].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Count', 'count')
        ]).round(4)
        
        return summary.reset_index()
    
    def rank_optimizers(self, df: pd.DataFrame, metric: str, ascending: bool = True) -> pd.DataFrame:
        """Rank optimizers based on average rank across instances."""
        # Pivot to get instances x optimizers
        pivot = df.pivot_table(index='instance', columns='optimizer', values=metric)
        
        # Calculate ranks (lower is better for Hide_The_Label, higher for Open_Race)
        ranks = pivot.rank(axis=1, ascending=ascending)
        
        # Calculate average rank for each optimizer
        avg_ranks = ranks.mean().sort_values()
        
        rank_df = pd.DataFrame({
            'Optimizer': avg_ranks.index,
            'Average_Rank': avg_ranks.values,
            'Rank_Position': range(1, len(avg_ranks) + 1)
        })
        
        return rank_df
    
    def analyze_setting(self, setting_path: str, experiment_type: str) -> Dict[str, Any]:
        """Perform complete analysis for a given setting."""
        print(f"\nAnalyzing: {setting_path}")
        
        # Load data
        files_data = self.load_json_files(setting_path)
        
        if not files_data:
            print(f"No JSON files found in {setting_path}")
            return None
        
        if experiment_type == 'Hide_The_Label':
            df = self.extract_hide_the_label_data(files_data)
            metric = 'steps_to_target'
            ascending = True  # Lower is better
        else:  # Open_Race
            df = self.extract_open_race_data(files_data)
            metric = 'final_best'
            ascending = False  # Higher is better
        
        if df.empty:
            print(f"WARNING: No competition data found in {setting_path}")
            print(f"         Files may have aggregated format without raw competition data.")
            return None
        
        print(f"Loaded {len(df)} records")
        print(f"Optimizers: {df['optimizer'].nunique()}")
        print(f"Instances: {df['instance'].nunique()}")
        print(f"Datasets: {df['dataset'].nunique()}")
        print(f"Batches: {df['batch'].nunique()}")
        
        # Perform analyses
        results = {
            'dataframe': df,
            'metric': metric,
            'ascending': ascending
        }
        
        # 1. Friedman test
        friedman_stat, friedman_p = self.friedman_test(df, metric)
        results['friedman'] = {
            'statistic': friedman_stat,
            'p_value': friedman_p
        }
        
        # 2. Optimizer rankings
        results['rankings'] = self.rank_optimizers(df, metric, ascending)
        
        # 3. Summary statistics
        results['summary_stats'] = self.optimizer_summary_statistics(df, metric)
        
        # 4. Pairwise comparisons
        pairwise = self.pairwise_comparisons(df, metric)
        results['pairwise'] = self.holm_bonferroni_correction(pairwise)
        
        # 5. Dataset-level analysis
        dataset_agg = self.dataset_level_analysis(df, metric)
        friedman_stat_ds, friedman_p_ds = self.friedman_test(dataset_agg, metric)
        results['dataset_level'] = {
            'data': dataset_agg,
            'friedman': {
                'statistic': friedman_stat_ds,
                'p_value': friedman_p_ds
            },
            'rankings': self.rank_optimizers(dataset_agg, metric, ascending)
        }
        
        return results
    
    def generate_latex_table(self, results: Dict[str, Any], title: str, 
                            experiment_type: str) -> str:
        """Generate LaTeX table from results."""
        metric = results['metric']
        metric_name = "Steps to Target" if experiment_type == "Hide_The_Label" else "Best Value"
        
        latex = f"\\begin{{table}}[ht]\n"
        latex += f"\\centering\n"
        latex += f"\\caption{{{title}}}\n"
        latex += f"\\label{{tab:{title.lower().replace(' ', '_')}}}\n"
        latex += f"\\begin{{tabular}}{{l" + "r" * 6 + "}}\n"
        latex += f"\\hline\n"
        latex += f"Optimizer & Mean & Median & Std & Min & Max & Count \\\\\n"
        latex += f"\\hline\n"
        
        # Sort by rank
        rankings = results['rankings']
        summary = results['summary_stats']
        
        for idx, row in rankings.iterrows():
            opt = row['Optimizer']
            stats = summary[summary['optimizer'] == opt].iloc[0]
            
            latex += f"{opt} & "
            latex += f"{stats['Mean']:.4f} & "
            latex += f"{stats['Median']:.4f} & "
            latex += f"{stats['Std']:.4f} & "
            latex += f"{stats['Min']:.4f} & "
            latex += f"{stats['Max']:.4f} & "
            latex += f"{int(stats['Count'])} \\\\\n"
        
        latex += f"\\hline\n"
        latex += f"\\end{{tabular}}\n"
        latex += f"\\end{{table}}\n"
        
        return latex
    
    def generate_markdown_table(self, results: Dict[str, Any], title: str, 
                               experiment_type: str) -> str:
        """Generate Markdown table from results."""
        metric = results['metric']
        
        # Summary Statistics Table
        md = f"\n## {title}\n\n"
        md += f"### Summary Statistics\n\n"
        
        summary = results['summary_stats']
        rankings = results['rankings']
        
        # Rename columns to match
        summary_renamed = summary.rename(columns={'optimizer': 'Optimizer'})
        
        # Merge rankings with summary
        merged = summary_renamed.merge(rankings[['Optimizer', 'Average_Rank', 'Rank_Position']], 
                              on='Optimizer', how='left')
        merged = merged.sort_values('Rank_Position')
        
        md += "| Rank | Optimizer | Mean | Median | Std | Min | Max | Count | Avg Rank |\n"
        md += "|------|-----------|------|--------|-----|-----|-----|-------|----------|\n"
        
        for idx, row in merged.iterrows():
            md += f"| {int(row['Rank_Position'])} | {row['Optimizer']} | "
            md += f"{row['Mean']:.4f} | {row['Median']:.4f} | {row['Std']:.4f} | "
            md += f"{row['Min']:.4f} | {row['Max']:.4f} | {int(row['Count'])} | "
            md += f"{row['Average_Rank']:.2f} |\n"
        
        # Friedman Test Results
        md += f"\n### Friedman Test (Instance-Level)\n\n"
        friedman = results['friedman']
        md += f"- **Statistic**: {friedman['statistic']:.4f}\n"
        md += f"- **P-Value**: {friedman['p_value']:.6f}\n"
        
        if friedman['p_value'] < 0.05:
            md += f"- **Result**: Significant differences detected (α = 0.05)\n"
        else:
            md += f"- **Result**: No significant differences detected (α = 0.05)\n"
        
        # Dataset-Level Friedman Test
        if 'dataset_level' in results:
            md += f"\n### Friedman Test (Dataset-Level)\n\n"
            ds_friedman = results['dataset_level']['friedman']
            md += f"- **Statistic**: {ds_friedman['statistic']:.4f}\n"
            md += f"- **P-Value**: {ds_friedman['p_value']:.6f}\n"
            
            if ds_friedman['p_value'] < 0.05:
                md += f"- **Result**: Significant differences detected (α = 0.05)\n"
            else:
                md += f"- **Result**: No significant differences detected (α = 0.05)\n"
        
        # Top Pairwise Comparisons
        md += f"\n### Top 20 Pairwise Comparisons (Holm-Bonferroni Corrected)\n\n"
        pairwise = results['pairwise'].head(20)
        
        md += "| Optimizer A | Optimizer B | HL Median | P-Value | Adj. P-Value | 95% CI | Significant |\n"
        md += "|-------------|-------------|-----------|---------|--------------|--------|-------------|\n"
        
        for idx, row in pairwise.iterrows():
            sig = "*" if row['Adjusted_P_Value'] < 0.05 else ""
            md += f"| {row['Optimizer_A']} | {row['Optimizer_B']} | "
            md += f"{row['HL_Median']:.4f} | {row['P_Value']:.6f} | "
            md += f"{row['Adjusted_P_Value']:.6f} | "
            md += f"[{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}] | {sig} |\n"
        
        return md
    
    def generate_csv_tables(self, results: Dict[str, Any], output_prefix: str):
        """Generate CSV files for each analysis component."""
        # Summary statistics
        summary_renamed = results['summary_stats'].rename(columns={'optimizer': 'Optimizer'})
        summary_renamed.merge(
            results['rankings'][['Optimizer', 'Average_Rank', 'Rank_Position']], 
            on='Optimizer'
        ).sort_values('Rank_Position').to_csv(f"{output_prefix}_summary.csv", index=False)
        
        # Pairwise comparisons
        results['pairwise'].to_csv(f"{output_prefix}_pairwise.csv", index=False)
        
        # Dataset-level rankings
        if 'dataset_level' in results:
            results['dataset_level']['rankings'].to_csv(
                f"{output_prefix}_dataset_rankings.csv", index=False
            )

def main():
    """Main analysis function.

    Paths default to the bundled ``Result_Official/hiddenfrac95`` tree at the
    repository root. Override with the ``AL_RESULTS_DIR`` environment variable
    if your post-processed results live elsewhere.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_path = os.environ.get(
        "AL_RESULTS_DIR",
        os.path.join(repo_root, "Result_Official", "hiddenfrac95"),
    )
    project_root = repo_root

    settings = [
        (f'{base_path}/Hard_Mode/Hide_The_Label', 'Hide_The_Label', 'Hard Mode - Hide The Label'),
        (f'{base_path}/Hard_Mode/Open_Race', 'Open_Race', 'Hard Mode - Open Race'),
        (f'{base_path}/Regular_Mode/Hide_The_Label', 'Hide_The_Label', 'Regular Mode - Hide The Label'),
        (f'{base_path}/Regular_Mode/Open_Race', 'Open_Race', 'Regular Mode - Open Race'),
    ]
    
    analyzer = StatisticalAnalyzer(base_path)
    all_results = {}
    
    # Perform analysis for each setting
    for setting_path, exp_type, title in settings:
        results = analyzer.analyze_setting(setting_path, exp_type)
        
        if results:
            all_results[title] = results
    
    # Generate output files
    output_dir = os.environ.get("AL_OUTPUT_DIR", os.path.join(repo_root, "Statistical_Analysis_Output"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Markdown report
    md_output = "# Statistical Analysis Tables\n"
    md_output += "\nGenerated: 2025-10-16\n"
    md_output += "\nThis report contains comprehensive statistical analysis for all experimental settings.\n"
    
    for title, results in all_results.items():
        exp_type = 'Hide_The_Label' if 'Hide' in title else 'Open_Race'
        md_output += analyzer.generate_markdown_table(results, title, exp_type)
        
        # Generate CSV files
        prefix = title.lower().replace(' ', '_').replace('-', '')
        csv_prefix = os.path.join(output_dir, f"statistical_analysis_{prefix}")
        analyzer.generate_csv_tables(results, csv_prefix)
    
    # Save markdown report
    md_file = os.path.join(output_dir, "STATISTICAL_ANALYSIS_TABLES.md")
    with open(md_file, 'w') as f:
        f.write(md_output)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"{'='*80}")
    print(f"\nMarkdown report saved to: {md_file}")
    print(f"\nCSV files saved with prefix: statistical_analysis_*")
    print(f"\nFiles generated:")
    for title in all_results.keys():
        prefix = title.lower().replace(' ', '_').replace('-', '')
        print(f"  - statistical_analysis_{prefix}_summary.csv")
        print(f"  - statistical_analysis_{prefix}_pairwise.csv")
        print(f"  - statistical_analysis_{prefix}_dataset_rankings.csv")

if __name__ == "__main__":
    main()
