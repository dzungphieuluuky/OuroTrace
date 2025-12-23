"""
Enhanced Metrics Module with Holistic Evaluation Analysis
Includes reasoning primitives (depth-k variable assignment) analysis
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration for comparison"""
    name: str
    size_b: float
    base_layers: int
    ut_steps: int
    effective_depth: int
    
    @property
    def flops_ratio(self) -> float:
        return self.effective_depth
    
    @property
    def param_ratio(self) -> float:
        return 1.0


class EnhancedOuroMetrics:
    """
    Extended metrics including holistic evaluation (reasoning primitives).
    """
    
    def __init__(self):
        self.results_cache = []
        self.holistic_cache = []
    
    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Add main experiment results"""
        self.results_cache.extend(results)
    
    def add_holistic_results(self, holistic_results: List[Dict[str, Any]]) -> None:
        """Add holistic evaluation results"""
        self.holistic_cache.extend(holistic_results)
    
    # =========================================================================
    # HOLISTIC EVALUATION METRICS
    # =========================================================================
    
    def compute_reasoning_primitive_accuracy(
        self,
        holistic_results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Analyze accuracy on reasoning primitives (depth-k variable assignment).
        
        Returns breakdown by:
        - Depth level (0, 1)
        - Variant (code, math, equation)
        - UT steps
        """
        if holistic_results is None:
            holistic_results = self.holistic_cache
        
        if not holistic_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(holistic_results)
        
        # Filter to reasoning primitives only
        df_primitives = df[df['task_category'] == 'Reasoning Primitive'].copy()
        
        if df_primitives.empty:
            return pd.DataFrame()
        
        # Extract depth and variant from task_name
        # Format: "var_assign_depth_{depth}_{variant}"
        df_primitives['depth'] = df_primitives['task_name'].str.extract(r'depth_(\d+)')[0].astype(int)
        df_primitives['variant'] = df_primitives['task_name'].str.extract(r'depth_\d+_(\w+)')[0]
        
        # Compute accuracy by depth, variant, and UT steps
        accuracy_breakdown = df_primitives.groupby(['depth', 'variant', 'ut_steps']).agg({
            'is_correct': ['mean', 'std', 'count']
        }).reset_index()
        
        accuracy_breakdown.columns = ['depth', 'variant', 'ut_steps', 'accuracy', 'std', 'n_samples']
        accuracy_breakdown['accuracy_pct'] = accuracy_breakdown['accuracy'] * 100
        
        return accuracy_breakdown
    
    def compute_depth_generalization(
        self,
        holistic_results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Analyze how accuracy changes from depth-0 to depth-1.
        
        Key metric: Does the model generalize to one-level indirection?
        """
        if holistic_results is None:
            holistic_results = self.holistic_cache
        
        if not holistic_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(holistic_results)
        df_primitives = df[df['task_category'] == 'Reasoning Primitive'].copy()
        
        if df_primitives.empty:
            return pd.DataFrame()
        
        df_primitives['depth'] = df_primitives['task_name'].str.extract(r'depth_(\d+)')[0].astype(int)
        df_primitives['variant'] = df_primitives['task_name'].str.extract(r'depth_\d+_(\w+)')[0]
        
        # Compare depth-0 vs depth-1 accuracy
        depth_comparison = df_primitives.groupby(['variant', 'depth', 'ut_steps']).agg({
            'is_correct': 'mean'
        }).reset_index()
        
        depth_comparison['accuracy_pct'] = depth_comparison['is_correct'] * 100
        
        # Pivot to show depth-0 vs depth-1 side by side
        pivot = depth_comparison.pivot_table(
            values='accuracy_pct',
            index=['variant', 'ut_steps'],
            columns='depth'
        ).reset_index()
        
        if 0 in pivot.columns and 1 in pivot.columns:
            pivot.columns.name = None
            pivot.rename(columns={0: 'depth_0_acc', 1: 'depth_1_acc'}, inplace=True)
            pivot['generalization_gap'] = pivot['depth_0_acc'] - pivot['depth_1_acc']
        
        return pivot
    
    def compute_variant_comparison(
        self,
        holistic_results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Compare accuracy across different prompt variants (code, math, equation).
        
        Shows format robustness.
        """
        if holistic_results is None:
            holistic_results = self.holistic_cache
        
        if not holistic_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(holistic_results)
        df_primitives = df[df['task_category'] == 'Reasoning Primitive'].copy()
        
        if df_primitives.empty:
            return pd.DataFrame()
        
        df_primitives['variant'] = df_primitives['task_name'].str.extract(r'depth_\d+_(\w+)')[0]
        
        variant_acc = df_primitives.groupby(['variant', 'ut_steps']).agg({
            'is_correct': ['mean', 'std', 'count']
        }).reset_index()
        
        variant_acc.columns = ['variant', 'ut_steps', 'accuracy', 'std', 'n_samples']
        variant_acc['accuracy_pct'] = variant_acc['accuracy'] * 100
        
        return variant_acc
    
    def compute_holistic_vs_main_comparison(
        self,
        main_results: Optional[List[Dict]] = None,
        holistic_results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Compare accuracy on main tasks vs reasoning primitives.
        
        Shows whether model performs differently on standard tasks vs primitives.
        """
        if main_results is None:
            main_results = self.results_cache
        if holistic_results is None:
            holistic_results = self.holistic_cache
        
        if not main_results or not holistic_results:
            return pd.DataFrame()
        
        df_main = pd.DataFrame(main_results)
        df_holistic = pd.DataFrame(holistic_results)
        
        # Main tasks accuracy
        main_acc = df_main.groupby('ut_steps')['is_correct'].mean().reset_index()
        main_acc.columns = ['ut_steps', 'main_accuracy']
        
        # Reasoning primitives accuracy
        df_primitives = df_holistic[df_holistic['task_category'] == 'Reasoning Primitive']
        if not df_primitives.empty:
            holistic_acc = df_primitives.groupby('ut_steps')['is_correct'].mean().reset_index()
            holistic_acc.columns = ['ut_steps', 'holistic_accuracy']
            
            # Merge
            comparison = main_acc.merge(holistic_acc, on='ut_steps', how='outer')
            comparison['main_accuracy_pct'] = comparison['main_accuracy'] * 100
            comparison['holistic_accuracy_pct'] = comparison['holistic_accuracy'] * 100
            comparison['gap'] = comparison['main_accuracy_pct'] - comparison['holistic_accuracy_pct']
            
            return comparison
        
        return pd.DataFrame()
    
    # =========================================================================
    # MAIN EXPERIMENT METRICS (keeping all original methods)
    # =========================================================================
    
    def compute_accuracy_by_ut_steps(
        self, 
        results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Accuracy vs UT Steps (Paper Figure 2, 3)"""
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        accuracy_by_ut = df.groupby(['task_type', 'ut_steps']).agg({
            'is_correct': ['mean', 'std', 'count']
        }).reset_index()
        
        accuracy_by_ut.columns = ['task_type', 'ut_steps', 'accuracy', 'std', 'n_samples']
        accuracy_by_ut['accuracy_pct'] = accuracy_by_ut['accuracy'] * 100
        
        return accuracy_by_ut
    
    def compute_depth_efficiency(
        self,
        results: Optional[List[Dict]] = None,
        model_configs: Optional[Dict[str, ModelConfig]] = None
    ) -> pd.DataFrame:
        """Depth Efficiency (Paper Claim 1)"""
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        depth_eff = df.groupby(['ut_steps']).agg({
            'is_correct': 'mean',
            'generation_time': 'mean'
        }).reset_index()
        
        if model_configs:
            depth_eff['effective_depth'] = depth_eff['ut_steps'].map(
                lambda x: model_configs.get(f'ut_{x}', ModelConfig('', 1.4, 24, x, 24*x)).effective_depth
            )
            depth_eff['depth_efficiency'] = depth_eff['is_correct'] / depth_eff['effective_depth']
        else:
            depth_eff['effective_depth'] = depth_eff['ut_steps'] * 24
            depth_eff['depth_efficiency'] = depth_eff['is_correct'] / depth_eff['effective_depth']
        
        depth_eff['accuracy_pct'] = depth_eff['is_correct'] * 100
        
        return depth_eff
    
    def compute_parameter_efficiency(
        self,
        results: Optional[List[Dict]] = None,
        model_size_b: float = 1.4
    ) -> pd.DataFrame:
        """Parameter Efficiency (Paper Table 1)"""
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        param_eff = df.groupby('ut_steps').agg({
            'is_correct': 'mean'
        }).reset_index()
        
        param_eff['model_size_b'] = model_size_b
        param_eff['param_efficiency'] = param_eff['is_correct'] / param_eff['model_size_b']
        param_eff['accuracy_pct'] = param_eff['is_correct'] * 100
        
        return param_eff
    
    def compute_throughput_efficiency(
        self,
        results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Token Efficiency"""
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        throughput = df.groupby('ut_steps').agg({
            'is_correct': 'mean',
            'generation_time': 'mean',
            'generated_tokens': 'mean'
        }).reset_index()
        
        throughput['tokens_per_sec'] = throughput['generated_tokens'] / throughput['generation_time']
        throughput['accuracy_pct'] = throughput['is_correct'] * 100
        throughput['accuracy_per_sec'] = throughput['is_correct'] / throughput['generation_time']
        
        return throughput
    
    def compute_difficulty_scaling(
        self,
        results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Step-wise Accuracy by difficulty"""
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        if 'difficulty' not in df.columns:
            return pd.DataFrame()
        
        difficulty_acc = df.groupby(['task_type', 'difficulty', 'ut_steps']).agg({
            'is_correct': ['mean', 'count']
        }).reset_index()
        
        difficulty_acc.columns = ['task_type', 'difficulty', 'ut_steps', 'accuracy', 'n_samples']
        difficulty_acc['accuracy_pct'] = difficulty_acc['accuracy'] * 100
        
        return difficulty_acc
    
    # =========================================================================
    # ENHANCED PLOTTING WITH HOLISTIC RESULTS
    # =========================================================================
    
    def generate_enhanced_plots(
        self,
        main_results: Optional[List[Dict]] = None,
        holistic_results: Optional[List[Dict]] = None,
        save_dir: str = "./plots"
    ) -> None:
        """
        Generate comprehensive plots including holistic evaluation.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if main_results is None:
            main_results = self.results_cache
        if holistic_results is None:
            holistic_results = self.holistic_cache
        
        # Plot 1: Main Tasks Accuracy
        if main_results:
            acc_by_ut = self.compute_accuracy_by_ut_steps(main_results)
            
            plt.figure(figsize=(10, 6))
            for task in acc_by_ut['task_type'].unique():
                task_data = acc_by_ut[acc_by_ut['task_type'] == task]
                plt.plot(task_data['ut_steps'], task_data['accuracy_pct'], 
                        marker='o', label=task, linewidth=2)
            
            plt.xlabel('UT Steps', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Main Tasks: Accuracy vs UT Steps', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{save_dir}/main_accuracy_vs_ut.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Reasoning Primitives by Depth
        if holistic_results:
            primitive_acc = self.compute_reasoning_primitive_accuracy(holistic_results)
            
            if not primitive_acc.empty:
                plt.figure(figsize=(12, 6))
                
                for (depth, variant), group in primitive_acc.groupby(['depth', 'variant']):
                    plt.plot(group['ut_steps'], group['accuracy_pct'], 
                            marker='o', label=f'Depth-{depth} ({variant})', linewidth=2)
                
                plt.xlabel('UT Steps', fontsize=12)
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.title('Reasoning Primitives: Accuracy by Depth & Variant', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{save_dir}/primitives_by_depth.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot 3: Depth Generalization Gap
        if holistic_results:
            depth_gen = self.compute_depth_generalization(holistic_results)
            
            if not depth_gen.empty and 'generalization_gap' in depth_gen.columns:
                plt.figure(figsize=(10, 6))
                
                for variant in depth_gen['variant'].unique():
                    var_data = depth_gen[depth_gen['variant'] == variant]
                    plt.plot(var_data['ut_steps'], var_data['generalization_gap'], 
                            marker='s', label=variant, linewidth=2)
                
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.xlabel('UT Steps', fontsize=12)
                plt.ylabel('Depth-0 Acc - Depth-1 Acc (%)', fontsize=12)
                plt.title('Generalization Gap (Depth-0 vs Depth-1)', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{save_dir}/generalization_gap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot 4: Main vs Holistic Comparison
        if main_results and holistic_results:
            comparison = self.compute_holistic_vs_main_comparison(main_results, holistic_results)
            
            if not comparison.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(comparison['ut_steps'], comparison['main_accuracy_pct'], 
                        marker='o', label='Main Tasks', linewidth=2)
                plt.plot(comparison['ut_steps'], comparison['holistic_accuracy_pct'], 
                        marker='s', label='Reasoning Primitives', linewidth=2)
                
                plt.xlabel('UT Steps', fontsize=12)
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.title('Main Tasks vs Reasoning Primitives', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{save_dir}/main_vs_holistic.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"✅ Enhanced plots saved to {save_dir}/")
    
    def generate_comprehensive_summary(
        self,
        main_results: Optional[List[Dict]] = None,
        holistic_results: Optional[List[Dict]] = None,
        model_name: str = "Ouro"
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive summary including holistic results.
        """
        if main_results is None:
            main_results = self.results_cache
        if holistic_results is None:
            holistic_results = self.holistic_cache
        
        summary = {}
        
        # Main tasks summary
        if main_results:
            df_main = pd.DataFrame(main_results)
            main_summary = df_main.groupby(['task_type', 'ut_steps']).agg({
                'is_correct': 'mean',
                'generation_time': 'mean',
                'generated_tokens': 'mean'
            }).reset_index()
            
            main_summary.columns = ['Task', 'UT Steps', 'Accuracy', 'Time (s)', 'Tokens']
            main_summary['Accuracy'] = (main_summary['Accuracy'] * 100).round(2)
            main_summary['Time (s)'] = main_summary['Time (s)'].round(3)
            main_summary['Tokens'] = main_summary['Tokens'].round(1)
            main_summary['Model'] = model_name
            main_summary['Category'] = 'Main Tasks'
            
            summary['main_tasks'] = main_summary
        
        # Holistic summary
        if holistic_results:
            df_holistic = pd.DataFrame(holistic_results)
            df_primitives = df_holistic[df_holistic['task_category'] == 'Reasoning Primitive']
            
            if not df_primitives.empty:
                holistic_summary = df_primitives.groupby(['task_name', 'ut_steps']).agg({
                    'is_correct': 'mean'
                }).reset_index()
                
                holistic_summary.columns = ['Task', 'UT Steps', 'Accuracy']
                holistic_summary['Accuracy'] = (holistic_summary['Accuracy'] * 100).round(2)
                holistic_summary['Model'] = model_name
                holistic_summary['Category'] = 'Reasoning Primitives'
                
                summary['reasoning_primitives'] = holistic_summary
        
        # Combined summary
        if 'main_tasks' in summary and 'reasoning_primitives' in summary:
            # Simplified combined view
            combined = pd.concat([
                summary['main_tasks'][['Model', 'Category', 'Task', 'UT Steps', 'Accuracy']],
                summary['reasoning_primitives'][['Model', 'Category', 'Task', 'UT Steps', 'Accuracy']]
            ])
            summary['combined'] = combined
        
        return summary


# ==============================================================================
# ENHANCED ANALYSIS FUNCTION
# ==============================================================================

def analyze_experiment_results(
    results_folder: str,
    save_plots: bool = True,
    save_dir: str = "./plots"
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive analysis including holistic evaluation.
    
    Args:
        results: Main experiment results (n_ary, p_hop, igsm)
        holistic_results: Holistic evaluation results (reasoning primitives)
        model_name: Model name for labeling
        model_size_b: Model size in billions
        save_plots: Whether to generate plots
        save_dir: Directory to save plots
    
    Returns:
        Dict of all computed metrics
    """
    print(f"\n{'='*70}")
    print(f"📊 COMPREHENSIVE METRICS ANALYSIS")
    print(f"{'='*70}\n")
    
    metrics = EnhancedOuroMetrics()

        # --- Load results CSV ---
    all_latest_path = os.path.join(results_folder, "all.csv")
    if not os.path.exists(all_latest_path):
        print(f"❌ all.csv not found in {results_folder}")
        return {}
    holistic_path = os.path.join(results_folder, "holistic.csv")
    if not os.path.exists(holistic_path):
        print(f"⚠️ holistic.csv not found in {results_folder}, proceeding without holistic results")
        return {}
    
    all_result_df = pd.read_csv(all_latest_path)
    simple_reasoning_results = all_result_df.to_dict(orient="records")
    holistic_result_df = pd.read_csv(holistic_path)
    holistic_results = holistic_result_df.to_dict(orient="records")

    # --- Load config.json ---
    config_path = os.path.join(results_folder, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ config.json not found in {results_folder}")
        model_name = "Ouro"
        model_size_b = 1.4
    else:
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        # Try to extract model name and size
        model_config = config.get("MODEL", {})
        model_hf_path = model_config.get("path", None)
        model_name = model_hf_path.split("/")[-1] if model_hf_path else None

        if not model_name:
            model_name = model_config.get("path", "Ouro")

        if "1.4" in model_name:
            model_size_b = 1.4
        elif "2.6" in model_name:
            model_size_b = 2.6
        else:
            print("⚠️ Unable to determine model size from config, defaulting to 1.4B")
            model_size_b = model_config.get("size_b", 1.4)

    metrics.add_results(simple_reasoning_results)
    
    if holistic_results:
        metrics.add_holistic_results(holistic_results)
    
    analysis_results = {}
    
    # =========================================================================
    # MAIN TASK METRICS
    # =========================================================================
    print("="*70)
    print("MAIN TASK ANALYSIS")
    print("="*70 + "\n")
    
    # Metric 1: Accuracy vs UT Steps
    print("📈 Accuracy vs UT Steps:")
    acc_by_ut = metrics.compute_accuracy_by_ut_steps()
    analysis_results['accuracy_by_ut'] = acc_by_ut
    print(acc_by_ut.to_string(index=False))
    print()
    
    # Metric 2: Depth Efficiency
    print("📈 Depth Efficiency:")
    depth_eff = metrics.compute_depth_efficiency()
    analysis_results['depth_efficiency'] = depth_eff
    print(depth_eff.to_string(index=False))
    print()
    
    # Metric 3: Parameter Efficiency
    print("📈 Parameter Efficiency:")
    param_eff = metrics.compute_parameter_efficiency(model_size_b=model_size_b)
    analysis_results['param_efficiency'] = param_eff
    print(param_eff.to_string(index=False))
    print()
    
    # Metric 4: Throughput
    print("📈 Throughput Efficiency:")
    throughput = metrics.compute_throughput_efficiency()
    analysis_results['throughput'] = throughput
    print(throughput.to_string(index=False))
    print()
    
    # =========================================================================
    # HOLISTIC EVALUATION METRICS
    # =========================================================================
    if holistic_results:
        print("="*70)
        print("REASONING PRIMITIVES ANALYSIS")
        print("="*70 + "\n")
        
        # Metric 5: Reasoning Primitive Accuracy
        print("📈 Reasoning Primitives by Depth & Variant:")
        primitive_acc = metrics.compute_reasoning_primitive_accuracy()
        if not primitive_acc.empty:
            analysis_results['primitive_accuracy'] = primitive_acc
            print(primitive_acc.to_string(index=False))
            print()
        
        # Metric 6: Depth Generalization
        print("📈 Depth Generalization (Depth-0 vs Depth-1):")
        depth_gen = metrics.compute_depth_generalization()
        if not depth_gen.empty:
            analysis_results['depth_generalization'] = depth_gen
            print(depth_gen.to_string(index=False))
            print()
        
        # Metric 7: Variant Comparison
        print("📈 Accuracy by Variant:")
        variant_comp = metrics.compute_variant_comparison()
        if not variant_comp.empty:
            analysis_results['variant_comparison'] = variant_comp
            print(variant_comp.to_string(index=False))
            print()
        
        # Metric 8: Main vs Holistic
        print("📈 Main Tasks vs Reasoning Primitives:")
        comparison = metrics.compute_holistic_vs_main_comparison()
        if not comparison.empty:
            analysis_results['main_vs_holistic'] = comparison
            print(comparison.to_string(index=False))
            print()
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("="*70)
    print("COMPREHENSIVE SUMMARY")
    print("="*70 + "\n")
    
    summary_tables = metrics.generate_comprehensive_summary(
        main_results=simple_reasoning_results,
        holistic_results=holistic_results,
        model_name=model_name
    )
    
    for table_name, table_df in summary_tables.items():
        print(f"📋 {table_name.replace('_', ' ').title()}:")
        print(table_df.to_string(index=False))
        print()
        analysis_results[f'summary_{table_name}'] = table_df
    
    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    if save_plots:
        print("📊 Generating plots...")
        metrics.generate_enhanced_plots(
            main_results=simple_reasoning_results,
            holistic_results=holistic_results,
            save_dir=save_dir
        )
    
    print(f"{'='*70}\n")
    
    return analysis_results


class PaperComplianceChecker:
    """
    Check if your experiment setup matches paper methodology.
    """
    
    @staticmethod
    def check_task_alignment(task_types: List[str]) -> Dict[str, bool]:
        """Check if tasks match paper"""
        paper_tasks = {'n_ary', 'p_hop', 'igsm'}
        
        alignment = {
            'has_n_ary': 'n_ary' in task_types,
            'has_p_hop': 'p_hop' in task_types,
            'has_igsm': 'igsm' in task_types,
            'all_paper_tasks': set(task_types) == paper_tasks
        }
        
        return alignment
    
    @staticmethod
    def check_ut_steps_coverage(ut_steps_list: List[int]) -> Dict[str, Any]:
        """Check if UT steps range is sufficient"""
        paper_range = [1, 2, 4, 8]  # Typical from paper
        
        coverage = {
            'min_ut': min(ut_steps_list),
            'max_ut': max(ut_steps_list),
            'covers_baseline': 1 in ut_steps_list,
            'covers_paper_range': all(x in ut_steps_list for x in paper_range if x <= max(ut_steps_list)),
            'recommended_range': paper_range
        }
        
        return coverage
    
    @staticmethod
    def recommend_experiments(
        model_sizes: List[str],
        current_ut_steps: List[int]
    ) -> Dict[str, List]:
        """Recommend additional experiments for paper-style analysis"""
        
        recommendations = {
            'missing_ut_steps': [],
            'comparison_experiments': [],
            'ablation_studies': []
        }
        
        # Recommend UT steps
        ideal_ut_steps = [1, 2, 4, 8, 16]
        recommendations['missing_ut_steps'] = [
            ut for ut in ideal_ut_steps if ut not in current_ut_steps
        ]
        
        # Recommend comparisons
        if len(model_sizes) >= 2:
            recommendations['comparison_experiments'].append(
                "Iso-param comparison: Compare 1.4B vs 2.6B at same UT steps"
            )
        
        if '1.4b' in model_sizes and '1.4b-thinking' in model_sizes:
            recommendations['comparison_experiments'].append(
                "Base vs Thinking: Compare pretrained thinking vs regular 1.4B"
            )
        
        # Recommend ablations
        recommendations['ablation_studies'] = [
            "Vary max_new_tokens to see if model uses extra capacity",
            "Test with/without few-shot examples",
            "Compare greedy vs sampling decoding"
        ]
        
        return recommendations


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze experiment results with paper-aligned metrics.")
    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Path to the folder containing all.csv and config.json"
    )
    args = parser.parse_args()
    
    try:
        paper_metrics = analyze_experiment_results(args.results_folder)
        for metric_name, df in paper_metrics.items():
            if not df.empty:
                filename = os.path.join(args.results_folder, f"{metric_name}.csv")
                df.to_csv(filename, index=False)
                print(f"✅ Saved {metric_name} to {filename}")

    except Exception as e:
        print(f"⚠️ Paper metrics analysis failed: {e}")
    