"""
Metrics Module Aligned with Ouroboros Paper (arXiv:2502.17416)

Key metrics from the paper that you can measure:
1. Accuracy vs UT Steps (scaling curve)
2. Depth Efficiency (accuracy per effective depth)
3. Parameter Efficiency (accuracy per billion parameters)
4. Token Efficiency (tokens/sec throughput)
5. Step-wise Accuracy (accuracy by problem difficulty)
6. Iso-Param Comparison (1.4B vs 2.6B at different depths)
7. Reasoning Trace Quality (step correctness)
8. Early Exit Analysis (when model stops thinking)

Place this in: evaluation.py or metrics.py
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
    size_b: float  # Size in billions
    base_layers: int  # Base number of layers (k)
    ut_steps: int  # Loop count (L)
    effective_depth: int  # k * L
    
    @property
    def flops_ratio(self) -> float:
        """Approximate FLOPs ratio compared to non-looped model"""
        return self.effective_depth
    
    @property
    def param_ratio(self) -> float:
        """Parameter ratio (constant across UT steps for same base model)"""
        return 1.0


class OuroMetrics:
    """
    Compute metrics aligned with Ouroboros paper.
    
    Main comparisons from paper:
    - (k ⊗ L): k-layer model looped L times (your setup)
    - (k ⊗ 1): k-layer model, no looping (baseline)
    - (kL ⊗ 1): kL-layer model (iso-FLOP baseline)
    """
    
    def __init__(self):
        self.results_cache = []
    
    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Add experiment results to cache"""
        self.results_cache.extend(results)
    
    def compute_accuracy_by_ut_steps(
        self, 
        results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Metric 1: Accuracy vs UT Steps (Paper Figure 2, 3)
        
        Shows how accuracy scales with thinking depth.
        Key finding: Should show improvement with more UT steps.
        """
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
        """
        Metric 2: Depth Efficiency (Paper Claim 1)
        
        Accuracy per unit of effective depth.
        Shows whether looping is efficient compared to just adding layers.
        """
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        # Group by model and UT steps
        depth_eff = df.groupby(['ut_steps']).agg({
            'is_correct': 'mean',
            'generation_time': 'mean'
        }).reset_index()
        
        # If model configs provided, add depth info
        if model_configs:
            depth_eff['effective_depth'] = depth_eff['ut_steps'].map(
                lambda x: model_configs.get(f'ut_{x}', ModelConfig('', 1.4, 24, x, 24*x)).effective_depth
            )
            depth_eff['depth_efficiency'] = depth_eff['is_correct'] / depth_eff['effective_depth']
        else:
            # Use UT steps as proxy for depth
            depth_eff['effective_depth'] = depth_eff['ut_steps'] * 24  # Assume 24 base layers
            depth_eff['depth_efficiency'] = depth_eff['is_correct'] / depth_eff['effective_depth']
        
        depth_eff['accuracy_pct'] = depth_eff['is_correct'] * 100
        
        return depth_eff
    
    def compute_parameter_efficiency(
        self,
        results: Optional[List[Dict]] = None,
        model_size_b: float = 1.4
    ) -> pd.DataFrame:
        """
        Metric 3: Parameter Efficiency (Paper Table 1)
        
        Accuracy per billion parameters.
        Key: Same parameters, different depths should show different accuracy.
        """
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
        """
        Metric 4: Token Efficiency
        
        Tokens generated per second vs accuracy.
        Shows inference cost vs quality tradeoff.
        """
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
        """
        Metric 5: Step-wise Accuracy (Paper experiments on n-ary, p-hop)
        
        Accuracy by problem difficulty level.
        Shows if looping helps more on harder problems.
        """
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
    
    def compute_iso_param_comparison(
        self,
        results_1_4b: List[Dict],
        results_2_6b: List[Dict]
    ) -> pd.DataFrame:
        """
        Metric 6: Iso-Param Comparison (Paper Figure 1 concept)
        
        Compare 1.4B vs 2.6B at same UT steps.
        Shows parameter count vs depth tradeoff.
        """
        df_1_4 = pd.DataFrame(results_1_4b)
        df_1_4['model_size'] = '1.4B'
        
        df_2_6 = pd.DataFrame(results_2_6b)
        df_2_6['model_size'] = '2.6B'
        
        df_combined = pd.concat([df_1_4, df_2_6])
        
        comparison = df_combined.groupby(['model_size', 'task_type', 'ut_steps']).agg({
            'is_correct': 'mean',
            'generation_time': 'mean'
        }).reset_index()
        
        comparison['accuracy_pct'] = comparison['is_correct'] * 100
        
        return comparison
    
    def compute_reasoning_trace_quality(
        self,
        results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Metric 7: Reasoning Trace Quality (Paper Section on latent thoughts)
        
        Analyze step-by-step correctness in generated responses.
        Only works if you extract intermediate steps.
        """
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        if 'full_response' not in df.columns:
            return pd.DataFrame()
        
        trace_quality = []
        
        for _, row in df.iterrows():
            response = row.get('full_response', '')
            
            # Count steps in response
            step_count = response.count('[STEP')
            has_final = '[FINAL]' in response
            
            trace_quality.append({
                'task_type': row['task_type'],
                'ut_steps': row['ut_steps'],
                'is_correct': row['is_correct'],
                'step_count': step_count,
                'has_final': has_final,
                'response_length': len(response),
            })
        
        trace_df = pd.DataFrame(trace_quality)
        
        trace_summary = trace_df.groupby(['task_type', 'ut_steps']).agg({
            'is_correct': 'mean',
            'step_count': 'mean',
            'has_final': 'mean',
            'response_length': 'mean'
        }).reset_index()
        
        trace_summary['accuracy_pct'] = trace_summary['is_correct'] * 100
        
        return trace_summary
    
    def compute_early_exit_analysis(
        self,
        results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Metric 8: Early Exit Analysis (Paper mentions early stopping)
        
        Requires model to output actual loops used vs max loops.
        Only works if your model logs this.
        """
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        # Check if early exit data is available
        if 'actual_ut_steps' not in df.columns:
            print("⚠️ Early exit data not available. Requires model instrumentation.")
            return pd.DataFrame()
        
        exit_analysis = df.groupby('ut_steps').agg({
            'actual_ut_steps': 'mean',
            'is_correct': 'mean'
        }).reset_index()
        
        exit_analysis['exit_efficiency'] = exit_analysis['actual_ut_steps'] / exit_analysis['ut_steps']
        exit_analysis['accuracy_pct'] = exit_analysis['is_correct'] * 100
        
        return exit_analysis
    
    def generate_paper_style_plots(
        self,
        results: Optional[List[Dict]] = None,
        save_dir: str = "./plots"
    ) -> None:
        """
        Generate plots similar to paper figures.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if results is None:
            results = self.results_cache
        
        # Plot 1: Accuracy vs UT Steps by Task
        acc_by_ut = self.compute_accuracy_by_ut_steps(results)
        
        plt.figure(figsize=(10, 6))
        for task in acc_by_ut['task_type'].unique():
            task_data = acc_by_ut[acc_by_ut['task_type'] == task]
            plt.plot(task_data['ut_steps'], task_data['accuracy_pct'], 
                    marker='o', label=task, linewidth=2)
        
        plt.xlabel('UT Steps (Thinking Depth)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy vs Thinking Depth (Paper Figure 2 style)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/accuracy_vs_ut_steps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Depth Efficiency
        depth_eff = self.compute_depth_efficiency(results)
        
        plt.figure(figsize=(10, 6))
        plt.plot(depth_eff['effective_depth'], depth_eff['accuracy_pct'], 
                marker='s', linewidth=2, color='purple')
        plt.xlabel('Effective Depth', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Depth Efficiency (Paper Claim 1)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/depth_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to {save_dir}/")
    
    def generate_paper_style_table(
        self,
        results: Optional[List[Dict]] = None,
        model_name: str = "Ouro"
    ) -> pd.DataFrame:
        """
        Generate summary table similar to paper tables.
        """
        if results is None:
            results = self.results_cache
        
        df = pd.DataFrame(results)
        
        # Create summary table
        summary = df.groupby(['task_type', 'ut_steps']).agg({
            'is_correct': 'mean',
            'generation_time': 'mean',
            'generated_tokens': 'mean'
        }).reset_index()
        
        summary.columns = ['Task', 'UT Steps', 'Accuracy', 'Time (s)', 'Tokens']
        summary['Accuracy'] = (summary['Accuracy'] * 100).round(2)
        summary['Time (s)'] = summary['Time (s)'].round(3)
        summary['Tokens'] = summary['Tokens'].round(1)
        summary['Model'] = model_name
        
        # Reorder columns
        summary = summary[['Model', 'Task', 'UT Steps', 'Accuracy', 'Time (s)', 'Tokens']]
        
        return summary


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

def analyze_experiment_results(
    results: Any,
    model_name: str = "Ouro-1.4B",
    model_size_b: float = 1.4,
    save_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Unified analysis function.
    Accepts either a list of dicts (in-memory results) or a CSV file path.
    Returns: Dict[str, pd.DataFrame] with all metrics and summary tables.
    """
    print(f"\n{'='*70}")
    if isinstance(results, str):
        print(f"📊 PAPER-ALIGNED METRICS ANALYSIS (CSV: {results})")
        if not os.path.exists(results):
            print(f"❌ File not found: {results}")
            return {}
        df = pd.read_csv(results)
        stats_results = df.to_dict(orient="records")
    elif isinstance(results, list):
        print(f"📊 PAPER-ALIGNED METRICS ANALYSIS (In-memory results)")
        stats_results = results
    else:
        print("❌ Invalid input: must be a list of dicts or CSV file path.")
        return {}
    print(f"{'='*70}\n")

    metrics = OuroMetrics()
    metrics.add_results(stats_results)

    analysis_results = {}

    # Metric 1: Accuracy vs UT Steps
    print("📈 Computing Accuracy vs UT Steps...")
    acc_by_ut = metrics.compute_accuracy_by_ut_steps()
    analysis_results['accuracy_by_ut'] = acc_by_ut
    print(acc_by_ut.to_string(index=False))
    print()

    # Metric 2: Depth Efficiency
    print("📈 Computing Depth Efficiency...")
    depth_eff = metrics.compute_depth_efficiency()
    analysis_results['depth_efficiency'] = depth_eff
    print(depth_eff.to_string(index=False))
    print()

    # Metric 3: Parameter Efficiency
    print("📈 Computing Parameter Efficiency...")
    param_eff = metrics.compute_parameter_efficiency(model_size_b=model_size_b)
    analysis_results['param_efficiency'] = param_eff
    print(param_eff.to_string(index=False))
    print()

    # Metric 4: Throughput Efficiency
    print("📈 Computing Throughput Efficiency...")
    throughput = metrics.compute_throughput_efficiency()
    analysis_results['throughput'] = throughput
    print(throughput.to_string(index=False))
    print()

    # Metric 7: Reasoning Trace Quality
    print("📈 Computing Reasoning Trace Quality...")
    trace_quality = metrics.compute_reasoning_trace_quality()
    if not trace_quality.empty:
        analysis_results['trace_quality'] = trace_quality
        print(trace_quality.to_string(index=False))
        print()

    # Generate summary table
    print("📋 Generating Paper-Style Summary Table...")
    summary_table = metrics.generate_paper_style_table(model_name=model_name)
    analysis_results['summary_table'] = summary_table
    print(summary_table.to_string(index=False))
    print()

    # Generate plots
    if save_plots:
        print("📊 Generating plots...")
        metrics.generate_paper_style_plots()
        print("✅ Plots saved to ./plots/")
        print()

    print(f"{'='*70}\n")

    return analysis_results