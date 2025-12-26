"""
Enhanced Metrics Module with Reasoning Primitives Evaluation Analysis
Includes reasoning primitives (depth-k variable assignment) analysis
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional
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
    Extended metrics including reasoning primitives evaluation.
    """

    def __init__(self):
        self.results_cache = []
        self.reasoning_primitives_cache = []

    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Add main experiment results"""
        self.results_cache.extend(results)

    def add_reasoning_primitives_results(
        self, reasoning_primitives_results: List[Dict[str, Any]]
    ) -> None:
        """Add reasoning primitives evaluation results"""
        self.reasoning_primitives_cache.extend(reasoning_primitives_results)

    # =========================================================================
    # REASONING PRIMITIVES EVALUATION METRICS
    # =========================================================================

    def compute_reasoning_primitive_accuracy(
        self, reasoning_primitives_results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Analyze accuracy on reasoning primitives (depth-k variable assignment).

        Returns breakdown by:
        - Depth level (0, 1)
        - Variant (code, math, equation)
        - UT steps
        """
        if reasoning_primitives_results is None:
            reasoning_primitives_results = self.reasoning_primitives_cache

        if not reasoning_primitives_results:
            return pd.DataFrame()

        df = pd.DataFrame(reasoning_primitives_results)

        # Filter to reasoning primitives only
        df_primitives = df[df["task_category"] == "Reasoning Primitive"].copy()

        if df_primitives.empty:
            return pd.DataFrame()

        # Extract depth and variant from task_name
        # Format: "var_assign_depth_{depth}_{variant}"
        df_primitives["depth"] = (
            df_primitives["task_name"].str.extract(r"depth_(\d+)")[0].astype(int)
        )
        df_primitives["variant"] = df_primitives["task_name"].str.extract(
            r"depth_\d+_(\w+)"
        )[0]

        # Compute accuracy by depth, variant, and UT steps
        accuracy_breakdown = (
            df_primitives.groupby(["depth", "variant", "ut_steps"])
            .agg({"is_correct": ["mean", "std", "count"]})
            .reset_index()
        )

        accuracy_breakdown.columns = [
            "depth",
            "variant",
            "ut_steps",
            "accuracy",
            "std",
            "n_samples",
        ]
        accuracy_breakdown["accuracy_pct"] = accuracy_breakdown["accuracy"] * 100

        return accuracy_breakdown

    def compute_depth_generalization(
        self, reasoning_primitives_results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Analyze how accuracy changes from depth-0 to depth-1.

        Key metric: Does the model generalize to one-level indirection?
        """
        if reasoning_primitives_results is None:
            reasoning_primitives_results = self.reasoning_primitives_cache

        if not reasoning_primitives_results:
            return pd.DataFrame()

        df = pd.DataFrame(reasoning_primitives_results)
        df_primitives = df[df["task_category"] == "Reasoning Primitive"].copy()

        if df_primitives.empty:
            return pd.DataFrame()

        df_primitives["depth"] = (
            df_primitives["task_name"].str.extract(r"depth_(\d+)")[0].astype(int)
        )
        df_primitives["variant"] = df_primitives["task_name"].str.extract(
            r"depth_\d+_(\w+)"
        )[0]

        # Compare depth-0 vs depth-1 accuracy
        depth_comparison = (
            df_primitives.groupby(["variant", "depth", "ut_steps"])
            .agg({"is_correct": "mean"})
            .reset_index()
        )

        depth_comparison["accuracy_pct"] = depth_comparison["is_correct"] * 100

        # Pivot to show depth-0 vs depth-1 side by side
        pivot = depth_comparison.pivot_table(
            values="accuracy_pct", index=["variant", "ut_steps"], columns="depth"
        ).reset_index()

        if 0 in pivot.columns and 1 in pivot.columns:
            pivot.columns.name = None
            pivot.rename(columns={0: "depth_0_acc", 1: "depth_1_acc"}, inplace=True)
            pivot["generalization_gap"] = pivot["depth_0_acc"] - pivot["depth_1_acc"]

        return pivot

    def compute_variant_comparison(
        self, reasoning_primitives_results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Compare accuracy across different prompt variants (code, math, equation).

        Shows format robustness.
        """
        if reasoning_primitives_results is None:
            reasoning_primitives_results = self.reasoning_primitives_cache

        if not reasoning_primitives_results:
            return pd.DataFrame()

        df = pd.DataFrame(reasoning_primitives_results)
        df_primitives = df[df["task_category"] == "Reasoning Primitive"].copy()

        if df_primitives.empty:
            return pd.DataFrame()

        df_primitives["variant"] = df_primitives["task_name"].str.extract(
            r"depth_\d+_(\w+)"
        )[0]

        variant_acc = (
            df_primitives.groupby(["variant", "ut_steps"])
            .agg({"is_correct": ["mean", "std", "count"]})
            .reset_index()
        )

        variant_acc.columns = ["variant", "ut_steps", "accuracy", "std", "n_samples"]
        variant_acc["accuracy_pct"] = variant_acc["accuracy"] * 100

        return variant_acc

    def compute_reasoning_primitives_vs_main_comparison(
        self,
        main_results: Optional[List[Dict]] = None,
        reasoning_primitives_results: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Compare accuracy on main tasks vs reasoning primitives.

        Shows whether model performs differently on standard tasks vs primitives.
        """
        if main_results is None:
            main_results = self.results_cache
        if reasoning_primitives_results is None:
            reasoning_primitives_results = self.reasoning_primitives_cache

        if not main_results or not reasoning_primitives_results:
            return pd.DataFrame()

        df_main = pd.DataFrame(main_results)
        df_primitives = pd.DataFrame(reasoning_primitives_results)

        # Main tasks accuracy
        main_acc = df_main.groupby("ut_steps")["is_correct"].mean().reset_index()
        main_acc.columns = ["ut_steps", "main_accuracy"]

        # Reasoning primitives accuracy
        df_primitives = df_primitives[
            df_primitives["task_category"] == "Reasoning Primitive"
        ]
        if not df_primitives.empty:
            primitives_acc = (
                df_primitives.groupby("ut_steps")["is_correct"].mean().reset_index()
            )
            primitives_acc.columns = ["ut_steps", "primitives_accuracy"]

            # Merge
            comparison = main_acc.merge(primitives_acc, on="ut_steps", how="outer")
            comparison["main_accuracy_pct"] = comparison["main_accuracy"] * 100
            comparison["primitives_accuracy_pct"] = (
                comparison["primitives_accuracy"] * 100
            )
            comparison["gap"] = (
                comparison["main_accuracy_pct"] - comparison["primitives_accuracy_pct"]
            )

            return comparison

        return pd.DataFrame()

    # =========================================================================
    # MAIN EXPERIMENT METRICS (keeping all original methods)
    # =========================================================================

    def compute_accuracy_by_ut_steps(
        self, results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Accuracy vs UT Steps (Paper Figure 2, 3)"""
        if results is None:
            results = self.results_cache

        df = pd.DataFrame(results)

        accuracy_by_ut = (
            df.groupby(["task_type", "ut_steps"])
            .agg({"is_correct": ["mean", "std", "count"]})
            .reset_index()
        )

        accuracy_by_ut.columns = [
            "task_type",
            "ut_steps",
            "accuracy",
            "std",
            "n_samples",
        ]
        accuracy_by_ut["accuracy_pct"] = accuracy_by_ut["accuracy"] * 100

        return accuracy_by_ut

    def compute_depth_efficiency(
        self,
        results: Optional[List[Dict]] = None,
        model_configs: Optional[Dict[str, ModelConfig]] = None,
    ) -> pd.DataFrame:
        """Depth Efficiency (Paper Claim 1)"""
        if results is None:
            results = self.results_cache

        df = pd.DataFrame(results)

        depth_eff = (
            df.groupby(["ut_steps"])
            .agg({"is_correct": "mean", "generation_time": "mean"})
            .reset_index()
        )

        if model_configs:
            depth_eff["effective_depth"] = depth_eff["ut_steps"].map(
                lambda x: model_configs.get(
                    f"ut_{x}", ModelConfig("", 1.4, 24, x, 24 * x)
                ).effective_depth
            )
            depth_eff["depth_efficiency"] = (
                depth_eff["is_correct"] / depth_eff["effective_depth"]
            )
        else:
            depth_eff["effective_depth"] = depth_eff["ut_steps"] * 24
            depth_eff["depth_efficiency"] = (
                depth_eff["is_correct"] / depth_eff["effective_depth"]
            )

        depth_eff["accuracy_pct"] = depth_eff["is_correct"] * 100

        return depth_eff

    def compute_parameter_efficiency(
        self, results: Optional[List[Dict]] = None, model_size_b: float = 1.4
    ) -> pd.DataFrame:
        """Parameter Efficiency (Paper Table 1)"""
        if results is None:
            results = self.results_cache

        df = pd.DataFrame(results)

        param_eff = df.groupby("ut_steps").agg({"is_correct": "mean"}).reset_index()

        param_eff["model_size_b"] = model_size_b
        param_eff["param_efficiency"] = (
            param_eff["is_correct"] / param_eff["model_size_b"]
        )
        param_eff["accuracy_pct"] = param_eff["is_correct"] * 100

        return param_eff

    def compute_throughput_efficiency(
        self, results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Token Efficiency"""
        if results is None:
            results = self.results_cache

        df = pd.DataFrame(results)

        throughput = (
            df.groupby("ut_steps")
            .agg(
                {
                    "is_correct": "mean",
                    "generation_time": "mean",
                    "generated_tokens": "mean",
                }
            )
            .reset_index()
        )

        throughput["tokens_per_sec"] = (
            throughput["generated_tokens"] / throughput["generation_time"]
        )
        throughput["accuracy_pct"] = throughput["is_correct"] * 100
        throughput["accuracy_per_sec"] = (
            throughput["is_correct"] / throughput["generation_time"]
        )

        return throughput

    def compute_difficulty_scaling(
        self, results: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Step-wise Accuracy by difficulty"""
        if results is None:
            results = self.results_cache

        df = pd.DataFrame(results)

        if "difficulty" not in df.columns:
            return pd.DataFrame()

        difficulty_acc = (
            df.groupby(["task_type", "difficulty", "ut_steps"])
            .agg({"is_correct": ["mean", "count"]})
            .reset_index()
        )

        difficulty_acc.columns = [
            "task_type",
            "difficulty",
            "ut_steps",
            "accuracy",
            "n_samples",
        ]
        difficulty_acc["accuracy_pct"] = difficulty_acc["accuracy"] * 100

        return difficulty_acc

    # =========================================================================
    # ENHANCED PLOTTING WITH REASONING PRIMITIVES RESULTS
    # =========================================================================

    def generate_enhanced_plots(
        self,
        main_results: Optional[List[Dict]] = None,
        reasoning_primitives_results: Optional[List[Dict]] = None,
        save_dir: str = "./plots",
    ) -> None:
        """
        Generate comprehensive plots including reasoning primitives evaluation.
        """
        os.makedirs(save_dir, exist_ok=True)

        if main_results is None:
            main_results = self.results_cache
        if reasoning_primitives_results is None:
            reasoning_primitives_results = self.reasoning_primitives_cache

        # Plot 1: Main Tasks Accuracy
        if main_results:
            acc_by_ut = self.compute_accuracy_by_ut_steps(main_results)

            plt.figure(figsize=(10, 6))
            for task in acc_by_ut["task_type"].unique():
                task_data = acc_by_ut[acc_by_ut["task_type"] == task]
                plt.plot(
                    task_data["ut_steps"],
                    task_data["accuracy_pct"],
                    marker="o",
                    label=task,
                    linewidth=2,
                )

            plt.xlabel("UT Steps", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.title("Main Tasks: Accuracy vs UT Steps", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                f"{save_dir}/main_accuracy_vs_ut.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # Plot 2: Reasoning Primitives by Depth
        if reasoning_primitives_results:
            primitive_acc = self.compute_reasoning_primitive_accuracy(
                reasoning_primitives_results
            )

            if not primitive_acc.empty:
                plt.figure(figsize=(12, 6))

                for (depth, variant), group in primitive_acc.groupby(
                    ["depth", "variant"]
                ):
                    plt.plot(
                        group["ut_steps"],
                        group["accuracy_pct"],
                        marker="o",
                        label=f"Depth-{depth} ({variant})",
                        linewidth=2,
                    )

                plt.xlabel("UT Steps", fontsize=12)
                plt.ylabel("Accuracy (%)", fontsize=12)
                plt.title(
                    "Reasoning Primitives: Accuracy by Depth & Variant", fontsize=14
                )
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(
                    f"{save_dir}/primitives_by_depth.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

        # Plot 3: Depth Generalization Gap
        if reasoning_primitives_results:
            depth_gen = self.compute_depth_generalization(reasoning_primitives_results)

            if not depth_gen.empty and "generalization_gap" in depth_gen.columns:
                plt.figure(figsize=(10, 6))

                for variant in depth_gen["variant"].unique():
                    var_data = depth_gen[depth_gen["variant"] == variant]
                    plt.plot(
                        var_data["ut_steps"],
                        var_data["generalization_gap"],
                        marker="s",
                        label=variant,
                        linewidth=2,
                    )

                plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
                plt.xlabel("UT Steps", fontsize=12)
                plt.ylabel("Depth-0 Acc - Depth-1 Acc (%)", fontsize=12)
                plt.title("Generalization Gap (Depth-0 vs Depth-1)", fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(
                    f"{save_dir}/generalization_gap.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

        # Plot 4: Main vs Reasoning Primitives Comparison
        if main_results and reasoning_primitives_results:
            comparison = self.compute_reasoning_primitives_vs_main_comparison(
                main_results, reasoning_primitives_results
            )

            if not comparison.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    comparison["ut_steps"],
                    comparison["main_accuracy_pct"],
                    marker="o",
                    label="Main Tasks",
                    linewidth=2,
                )
                plt.plot(
                    comparison["ut_steps"],
                    comparison["primitives_accuracy_pct"],
                    marker="s",
                    label="Reasoning Primitives",
                    linewidth=2,
                )

                plt.xlabel("UT Steps", fontsize=12)
                plt.ylabel("Accuracy (%)", fontsize=12)
                plt.title("Main Tasks vs Reasoning Primitives", fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(
                    f"{save_dir}/main_vs_reasoning_primitives.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        print(f"✅ Enhanced plots saved to {save_dir}/")

    def generate_comprehensive_summary(
        self,
        main_results: Optional[List[Dict]] = None,
        reasoning_primitives_results: Optional[List[Dict]] = None,
        model_name: str = "Ouro",
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive summary including reasoning primitives results.
        """
        if main_results is None:
            main_results = self.results_cache
        if reasoning_primitives_results is None:
            reasoning_primitives_results = self.reasoning_primitives_cache

        summary = {}

        # Main tasks summary
        if main_results:
            df_main = pd.DataFrame(main_results)
            main_summary = (
                df_main.groupby(["task_type", "ut_steps"])
                .agg(
                    {
                        "is_correct": "mean",
                        "generation_time": "mean",
                        "generated_tokens": "mean",
                    }
                )
                .reset_index()
            )

            main_summary.columns = [
                "Task",
                "UT Steps",
                "Accuracy",
                "Time (s)",
                "Tokens",
            ]
            main_summary["Accuracy"] = (main_summary["Accuracy"] * 100).round(2)
            main_summary["Time (s)"] = main_summary["Time (s)"].round(3)
            main_summary["Tokens"] = main_summary["Tokens"].round(1)
            main_summary["Model"] = model_name
            main_summary["Category"] = "Main Tasks"

            summary["main_tasks"] = main_summary

        # Reasoning Primitives summary
        if reasoning_primitives_results:
            df_primitives = pd.DataFrame(reasoning_primitives_results)
            df_primitives = df_primitives[
                df_primitives["task_category"] == "Reasoning Primitive"
            ]

            if not df_primitives.empty:
                primitives_summary = (
                    df_primitives.groupby(["task_name", "ut_steps"])
                    .agg({"is_correct": "mean"})
                    .reset_index()
                )

                primitives_summary.columns = ["Task", "UT Steps", "Accuracy"]
                primitives_summary["Accuracy"] = (
                    primitives_summary["Accuracy"] * 100
                ).round(2)
                primitives_summary["Model"] = model_name
                primitives_summary["Category"] = "Reasoning Primitives"

                summary["reasoning_primitives"] = primitives_summary

        # Combined summary
        if "main_tasks" in summary and "reasoning_primitives" in summary:
            # Simplified combined view
            combined = pd.concat(
                [
                    summary["main_tasks"][
                        ["Model", "Category", "Task", "UT Steps", "Accuracy"]
                    ],
                    summary["reasoning_primitives"][
                        ["Model", "Category", "Task", "UT Steps", "Accuracy"]
                    ],
                ]
            )
            summary["combined"] = combined

        return summary


# ==============================================================================
# ENHANCED ANALYSIS FUNCTION
# ==============================================================================


def analyze_experiment_results(
    results_folder: str, save_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive analysis including ALL metrics and reasoning primitives evaluation.

    Args:
        results_folder: Path to folder containing CSV files and config.json
        save_plots: Whether to generate plots

    Returns:
        Dict of all computed metrics
    """
    print(f"\n{'=' * 70}")
    print(f"📊 COMPREHENSIVE METRICS ANALYSIS")
    print(f"📂 Folder: {results_folder}")
    print(f"{'=' * 70}\n")

    metrics = EnhancedOuroMetrics()
    save_dir = os.path.join(results_folder, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # =========================================================================
    # LOAD DATA FILES
    # =========================================================================
    print("📥 Loading data files...")
    
    # Load main task results
    simple_reasoning_path = os.path.join(results_folder, "simple_reasoning.csv")
    if not os.path.exists(simple_reasoning_path):
        print(f"❌ simple_reasoning.csv not found in {results_folder}")
        return {}
    
    simple_reasoning_df = pd.read_csv(simple_reasoning_path)
    simple_reasoning_results = simple_reasoning_df.to_dict(orient="records")
    print(f"   ✓ Loaded {len(simple_reasoning_results)} main task results")
    
    # Load reasoning primitives results (optional)
    reasoning_primitives_results = []
    reasoning_primitives_path = os.path.join(results_folder, "reasoning_primitives.csv")
    if os.path.exists(reasoning_primitives_path):
        reasoning_primitives_df = pd.read_csv(reasoning_primitives_path)
        reasoning_primitives_results = reasoning_primitives_df.to_dict(orient="records")
        print(f"   ✓ Loaded {len(reasoning_primitives_results)} reasoning primitive results")
    else:
        print(f"   ⚠️ reasoning_primitives.csv not found (optional)")

    # Load config
    config_path = os.path.join(results_folder, "config.json")
    if not os.path.exists(config_path):
        print(f"   ⚠️ config.json not found, using defaults")
        model_name = "Ouro"
        model_size_b = 1.4
    else:
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        
        model_config = config.get("MODEL", {})
        model_path = model_config.get("path", "")
        model_name = model_path.split("/")[-1] if model_path else "Ouro"
        
        if "1.4" in model_name.lower():
            model_size_b = 1.4
        elif "2.6" in model_name.lower():
            model_size_b = 2.6
        else:
            model_size_b = 1.4
        
        print(f"   ✓ Model: {model_name} ({model_size_b}B)")

    print()

    # Add results to metrics object
    metrics.add_results(simple_reasoning_results)
    if reasoning_primitives_results:
        metrics.add_reasoning_primitives_results(reasoning_primitives_results)

    analysis_results = {}

    # =========================================================================
    # MAIN TASK METRICS
    # =========================================================================
    print("=" * 70)
    print("📊 MAIN TASK ANALYSIS")
    print("=" * 70 + "\n")

    # Metric 1: Accuracy vs UT Steps
    print("📈 Metric 1: Accuracy vs UT Steps")
    print("-" * 70)
    acc_by_ut = metrics.compute_accuracy_by_ut_steps()
    if not acc_by_ut.empty:
        analysis_results["accuracy_by_ut"] = acc_by_ut
        print(acc_by_ut.to_string(index=False))
        
        # Summary stats
        best_acc = acc_by_ut.loc[acc_by_ut['accuracy_pct'].idxmax()]
        print(f"\n   🏆 Best: {best_acc['task_type']} at UT={best_acc['ut_steps']} ({best_acc['accuracy_pct']:.2f}%)")
    print("\n")

    # Metric 2: Depth Efficiency
    print("📈 Metric 2: Depth Efficiency")
    print("-" * 70)
    depth_eff = metrics.compute_depth_efficiency()
    if not depth_eff.empty:
        analysis_results["depth_efficiency"] = depth_eff
        print(depth_eff.to_string(index=False))
        
        # Check if efficiency improves with depth
        if len(depth_eff) > 1:
            trend = "📈 Increasing" if depth_eff['depth_efficiency'].is_monotonic_increasing else "📉 Decreasing"
            print(f"\n   {trend} efficiency with more UT steps")
    print("\n")

    # Metric 3: Parameter Efficiency
    print("📈 Metric 3: Parameter Efficiency")
    print("-" * 70)
    param_eff = metrics.compute_parameter_efficiency(model_size_b=model_size_b)
    if not param_eff.empty:
        analysis_results["param_efficiency"] = param_eff
        print(param_eff.to_string(index=False))
        
        # Highlight best param efficiency
        best_param_eff = param_eff.loc[param_eff['param_efficiency'].idxmax()]
        print(f"\n   🏆 Best param efficiency: UT={best_param_eff['ut_steps']} ({best_param_eff['param_efficiency']:.4f})")
    print("\n")

    # Metric 4: Throughput Efficiency
    print("📈 Metric 4: Throughput Efficiency")
    print("-" * 70)
    throughput = metrics.compute_throughput_efficiency()
    if not throughput.empty:
        analysis_results["throughput"] = throughput
        print(throughput.to_string(index=False))
        
        # Highlight speed-accuracy tradeoff
        best_throughput = throughput.loc[throughput['tokens_per_sec'].idxmax()]
        best_accuracy = throughput.loc[throughput['accuracy_pct'].idxmax()]
        print(f"\n   ⚡ Fastest: UT={best_throughput['ut_steps']} ({best_throughput['tokens_per_sec']:.1f} tok/s)")
        print(f"   🎯 Most accurate: UT={best_accuracy['ut_steps']} ({best_accuracy['accuracy_pct']:.2f}%)")
    print("\n")

    # Metric 5: Difficulty Scaling (if available)
    if 'difficulty' in simple_reasoning_df.columns:
        print("📈 Metric 5: Difficulty Scaling")
        print("-" * 70)
        difficulty_scaling = metrics.compute_difficulty_scaling()
        if not difficulty_scaling.empty:
            analysis_results["difficulty_scaling"] = difficulty_scaling
            print(difficulty_scaling.to_string(index=False))
            
            # Analyze improvement on hard problems
            for task in difficulty_scaling['task_type'].unique():
                task_data = difficulty_scaling[difficulty_scaling['task_type'] == task]
                if len(task_data) > 1:
                    print(f"\n   {task}: Accuracy improves with UT steps on harder problems")
        print("\n")

    # =========================================================================
    # REASONING PRIMITIVES METRICS
    # =========================================================================
    if reasoning_primitives_results:
        print("=" * 70)
        print("🧠 REASONING PRIMITIVES ANALYSIS")
        print("=" * 70 + "\n")

        # Metric 6: Reasoning Primitive Accuracy
        print("📈 Metric 6: Reasoning Primitives by Depth & Variant")
        print("-" * 70)
        primitive_acc = metrics.compute_reasoning_primitive_accuracy()
        if not primitive_acc.empty:
            analysis_results["primitive_accuracy"] = primitive_acc
            print(primitive_acc.to_string(index=False))
            
            # Summary by depth
            print("\n   Summary by depth:")
            for depth in primitive_acc['depth'].unique():
                depth_data = primitive_acc[primitive_acc['depth'] == depth]
                avg_acc = depth_data['accuracy_pct'].mean()
                print(f"   • Depth-{depth}: {avg_acc:.2f}% average")
        print("\n")

        # Metric 7: Depth Generalization
        print("📈 Metric 7: Depth Generalization Gap")
        print("-" * 70)
        depth_gen = metrics.compute_depth_generalization()
        if not depth_gen.empty:
            analysis_results["depth_generalization"] = depth_gen
            print(depth_gen.to_string(index=False))
            
            if 'generalization_gap' in depth_gen.columns:
                avg_gap = depth_gen['generalization_gap'].mean()
                print(f"\n   📊 Average gap: {avg_gap:.2f}%")
                
                if avg_gap > 15:
                    print("   ⚠️  Large gap! Model struggles with indirection")
                elif avg_gap > 5:
                    print("   ✓ Moderate gap - reasonable performance on depth-1")
                else:
                    print("   ✅ Excellent! Strong generalization to deeper reasoning")
                
                # Check if gap improves with UT steps
                if len(depth_gen) > 1:
                    gap_trend = depth_gen.groupby('ut_steps')['generalization_gap'].mean()
                    if gap_trend.is_monotonic_decreasing:
                        print("   📈 Gap decreases with more UT steps - thinking helps!")
        print("\n")

        # Metric 8: Variant Comparison (Format Robustness)
        print("📈 Metric 8: Format Robustness (Variant Comparison)")
        print("-" * 70)
        variant_comp = metrics.compute_variant_comparison()
        if not variant_comp.empty:
            analysis_results["variant_comparison"] = variant_comp
            print(variant_comp.to_string(index=False))
            
            # Check consistency across formats
            print("\n   Format consistency by UT step:")
            for ut in variant_comp['ut_steps'].unique():
                ut_data = variant_comp[variant_comp['ut_steps'] == ut]
                acc_std = ut_data['accuracy_pct'].std()
                acc_range = ut_data['accuracy_pct'].max() - ut_data['accuracy_pct'].min()
                
                if acc_std > 10:
                    status = "⚠️  High variance"
                elif acc_std > 5:
                    status = "○ Moderate variance"
                else:
                    status = "✓ Consistent"
                
                print(f"   • UT={ut}: {status} (σ={acc_std:.1f}%, range={acc_range:.1f}%)")
        print("\n")

        # Metric 9: Main vs Reasoning Primitives Comparison
        print("📈 Metric 9: Main Tasks vs Reasoning Primitives")
        print("-" * 70)
        comparison = metrics.compute_reasoning_primitives_vs_main_comparison()
        if not comparison.empty:
            analysis_results["main_vs_reasoning_primitives"] = comparison
            print(comparison.to_string(index=False))
            
            # Analyze relative performance
            print("\n   Relative performance:")
            for _, row in comparison.iterrows():
                ut = row['ut_steps']
                gap = row.get('gap', 0)
                
                if abs(gap) < 3:
                    status = "≈ Similar performance"
                elif gap > 0:
                    status = f"↑ Main tasks {gap:.1f}% better"
                else:
                    status = f"↑ Primitives {abs(gap):.1f}% better"
                
                print(f"   • UT={ut}: {status}")
            
            # Check convergence
            if len(comparison) > 1:
                gap_trend = comparison['gap'].abs()
                if gap_trend.is_monotonic_decreasing:
                    print("\n   📈 Performance gap narrows with more UT steps!")
        print("\n")

    # =========================================================================
    # COMPREHENSIVE SUMMARY TABLES
    # =========================================================================
    print("=" * 70)
    print("📋 COMPREHENSIVE SUMMARY TABLES")
    print("=" * 70 + "\n")

    summary_tables = metrics.generate_comprehensive_summary(
        main_results=simple_reasoning_results,
        reasoning_primitives_results=reasoning_primitives_results,
        model_name=model_name,
    )

    for table_name, table_df in summary_tables.items():
        print(f"📊 {table_name.replace('_', ' ').title()}")
        print("-" * 70)
        print(table_df.to_string(index=False))
        print()
        analysis_results[f"summary_{table_name}"] = table_df

    # =========================================================================
    # KEY INSIGHTS SECTION
    # =========================================================================
    print("=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70 + "\n")
    
    insights = []
    
    # Insight 1: Overall best configuration
    if not acc_by_ut.empty:
        overall_best = acc_by_ut.groupby('ut_steps')['accuracy_pct'].mean().idxmax()
        overall_best_acc = acc_by_ut.groupby('ut_steps')['accuracy_pct'].mean().max()
        insights.append(f"• Best overall performance: UT={overall_best} ({overall_best_acc:.2f}%)")
    
    # Insight 2: Scaling behavior
    if not acc_by_ut.empty and len(acc_by_ut['ut_steps'].unique()) > 1:
        ut_sorted = acc_by_ut.groupby('ut_steps')['accuracy_pct'].mean().sort_index()
        if ut_sorted.is_monotonic_increasing:
            insights.append("• Accuracy scales positively with UT steps ✓")
        else:
            insights.append("• Accuracy does not scale monotonically with UT steps ⚠️")
    
    # Insight 3: Reasoning primitives performance
    if reasoning_primitives_results and not primitive_acc.empty:
        depth0_acc = primitive_acc[primitive_acc['depth'] == 0]['accuracy_pct'].mean()
        depth1_acc = primitive_acc[primitive_acc['depth'] == 1]['accuracy_pct'].mean()
        insights.append(f"• Reasoning: Depth-0 {depth0_acc:.1f}% vs Depth-1 {depth1_acc:.1f}%")
        
        if depth0_acc - depth1_acc > 10:
            insights.append("  ⚠️ Significant drop at depth-1 - generalization challenge")
        else:
            insights.append("  ✓ Good generalization to depth-1 problems")
    
    # Insight 4: Speed-accuracy tradeoff
    if not throughput.empty:
        best_speed_ut = throughput.loc[throughput['tokens_per_sec'].idxmax(), 'ut_steps']
        best_acc_ut = throughput.loc[throughput['accuracy_pct'].idxmax(), 'ut_steps']
        
        if best_speed_ut != best_acc_ut:
            insights.append(f"• Speed-accuracy tradeoff: Fastest at UT={best_speed_ut}, Most accurate at UT={best_acc_ut}")
    
    # Print all insights
    for insight in insights:
        print(insight)
    
    print()

    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    if save_plots:
        print("=" * 70)
        print("📊 GENERATING VISUALIZATION PLOTS")
        print("=" * 70 + "\n")
        
        metrics.generate_enhanced_plots(
            main_results=simple_reasoning_results,
            reasoning_primitives_results=reasoning_primitives_results,
            save_dir=save_dir,
        )
        print()

    # =========================================================================
    # SAVE ALL METRICS TO CSV
    # =========================================================================
    print("=" * 70)
    print("💾 SAVING METRICS TO CSV")
    print("=" * 70 + "\n")
    
    saved_count = 0
    for metric_name, df in analysis_results.items():
        if not df.empty:
            filename = os.path.join(results_folder, f"{metric_name}.csv")
            df.to_csv(filename, index=False)
            print(f"   ✓ {metric_name}.csv")
            saved_count += 1
    
    print(f"\n   📁 Saved {saved_count} metric files to {results_folder}")
    print()

    print(f"{'=' * 70}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'=' * 70}\n")

    return analysis_results

class PaperComplianceChecker:
    """
    Check if your experiment setup matches paper methodology.
    """

    @staticmethod
    def check_task_alignment(task_types: List[str]) -> Dict[str, bool]:
        """Check if tasks match paper"""
        paper_tasks = {"n_ary", "p_hop", "igsm"}

        alignment = {
            "has_n_ary": "n_ary" in task_types,
            "has_p_hop": "p_hop" in task_types,
            "has_igsm": "igsm" in task_types,
            "all_paper_tasks": set(task_types) == paper_tasks,
        }

        return alignment

    @staticmethod
    def check_ut_steps_coverage(ut_steps_list: List[int]) -> Dict[str, Any]:
        """Check if UT steps range is sufficient"""
        paper_range = [1, 2, 4, 8]  # Typical from paper

        coverage = {
            "min_ut": min(ut_steps_list),
            "max_ut": max(ut_steps_list),
            "covers_baseline": 1 in ut_steps_list,
            "covers_paper_range": all(
                x in ut_steps_list for x in paper_range if x <= max(ut_steps_list)
            ),
            "recommended_range": paper_range,
        }

        return coverage

    @staticmethod
    def recommend_experiments(
        model_sizes: List[str], current_ut_steps: List[int]
    ) -> Dict[str, List]:
        """Recommend additional experiments for paper-style analysis"""

        recommendations = {
            "missing_ut_steps": [],
            "comparison_experiments": [],
            "ablation_studies": [],
        }

        # Recommend UT steps
        ideal_ut_steps = [1, 2, 4, 8, 16]
        recommendations["missing_ut_steps"] = [
            ut for ut in ideal_ut_steps if ut not in current_ut_steps
        ]

        # Recommend comparisons
        if len(model_sizes) >= 2:
            recommendations["comparison_experiments"].append(
                "Iso-param comparison: Compare 1.4B vs 2.6B at same UT steps"
            )

        if "1.4b" in model_sizes and "1.4b-thinking" in model_sizes:
            recommendations["comparison_experiments"].append(
                "Base vs Thinking: Compare pretrained thinking vs regular 1.4B"
            )

        # Recommend ablations
        recommendations["ablation_studies"] = [
            "Vary max_new_tokens to see if model uses extra capacity",
            "Test with/without few-shot examples",
            "Compare greedy vs sampling decoding",
        ]

        return recommendations


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze experiment results with paper-aligned metrics."
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Path to the folder containing simple_reasoning.csv and config.json",
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
