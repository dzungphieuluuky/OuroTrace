import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Configuration
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 10

FILE_PATH = "ouro_acc_20251215_061001.csv"


def load_data(filepath):
    """Load and prepare the dataset."""
    try:
        df = pd.read_csv(filepath)
        df["is_correct_int"] = df["is_correct"].astype(int)
        print(f"✓ Loaded {len(df)} test records")
        print(f"✓ Task types: {df['task_type'].unique().tolist()}")
        print(f"✓ Difficulty levels: {sorted(df['difficulty'].unique().tolist())}")
        return df
    except FileNotFoundError:
        print(f"✗ ERROR: Could not find '{filepath}'")
        return None


def print_summary_table(df, group_cols, metrics, title):
    """Print formatted summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    agg_dict = {
        col: ["mean", "count"] if col == "is_correct_int" else "mean" for col in metrics
    }

    summary = df.groupby(group_cols).agg(agg_dict).round(4)

    # Flatten columns
    summary.columns = [
        "_".join(str(c) for c in col).strip("_") for col in summary.columns.values
    ]

    # Rename for clarity
    rename_map = {
        "is_correct_int_mean": "Accuracy",
        "is_correct_int_count": "N",
        "generated_tokens_mean": "Avg_Tokens",
        "generation_time_mean": "Avg_Time_s",
    }
    summary = summary.rename(columns=rename_map)

    # Format output
    if "Accuracy" in summary.columns:
        summary["Accuracy"] = (summary["Accuracy"] * 100).apply(lambda x: f"{x:.1f}%")
    if "Avg_Tokens" in summary.columns:
        summary["Avg_Tokens"] = summary["Avg_Tokens"].apply(lambda x: f"{x:.0f}")
    if "Avg_Time_s" in summary.columns:
        summary["Avg_Time_s"] = summary["Avg_Time_s"].apply(lambda x: f"{x:.2f}")

    print(summary.to_string())
    print()


def plot_accuracy_comparison(df):
    """Accuracy comparison across task types and difficulty."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # By task type and difficulty
    summary = (
        df.groupby(["task_type", "difficulty"])["is_correct_int"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary["label"] = summary.apply(
        lambda x: f"{x['mean'] * 100:.1f}%\n(n={x['count']})", axis=1
    )

    pivot = summary.pivot(index="task_type", columns="difficulty", values="mean")
    pivot.plot(kind="bar", ax=ax1, colormap="viridis", width=0.8)
    ax1.set_title(
        "Accuracy by Task Type and Difficulty", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_xlabel("Task Type", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(title="Difficulty", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Overall by task type
    task_summary = (
        df.groupby("task_type").agg({"is_correct_int": ["mean", "count"]}).reset_index()
    )
    task_summary.columns = ["task_type", "accuracy", "count"]

    bars = ax2.bar(
        task_summary["task_type"],
        task_summary["accuracy"],
        color=sns.color_palette("viridis", len(task_summary)),
    )
    ax2.set_title("Overall Accuracy by Task Type", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_xlabel("Task Type", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, task_summary.itertuples())):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height * 100:.1f}%\n(n={row.count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_performance_scatter(df):
    """Analyze relationship between time, tokens, and correctness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Time vs Tokens colored by correctness
    for correct, color, label in [
        (True, "green", "Correct"),
        (False, "red", "Incorrect"),
    ]:
        mask = df["is_correct"] == correct
        for task in df["task_type"].unique():
            task_mask = mask & (df["task_type"] == task)
            ax1.scatter(
                df[task_mask]["generated_tokens"],
                df[task_mask]["generation_time"],
                c=color,
                alpha=0.6,
                s=60,
                label=f"{task} - {label}" if correct else None,
                marker="o" if task == df["task_type"].unique()[0] else "s",
            )

    ax1.set_title("Generation Time vs Tokens Generated", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Generated Tokens", fontsize=12)
    ax1.set_ylabel("Generation Time (seconds)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # Token efficiency by correctness
    df.boxplot(
        column="generated_tokens",
        by=["task_type", "is_correct"],
        ax=ax2,
        patch_artist=True,
    )
    ax2.set_title(
        "Token Distribution: Correct vs Incorrect", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Task Type - Correctness", fontsize=12)
    ax2.set_ylabel("Generated Tokens", fontsize=12)
    plt.suptitle("")  # Remove default title

    plt.tight_layout()
    plt.show()


def plot_ut_steps_analysis(df):
    """Analyze impact of UT steps if variable."""
    if df["ut_steps"].nunique() <= 1:
        print("[INFO] Skipping UT Steps analysis - only one unique value found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy by UT steps
    for task in df["task_type"].unique():
        task_data = df[df["task_type"] == task]
        steps_summary = (
            task_data.groupby("ut_steps").agg({"is_correct_int": "mean"}).reset_index()
        )
        ax1.plot(
            steps_summary["ut_steps"],
            steps_summary["is_correct_int"],
            marker="o",
            linewidth=2.5,
            label=task,
            markersize=8,
        )

    ax1.set_title("Accuracy vs UT Steps", fontsize=14, fontweight="bold")
    ax1.set_xlabel("UT Steps", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(title="Task Type")
    ax1.grid(True, alpha=0.3)

    # Token usage by UT steps
    df.boxplot(column="generated_tokens", by="ut_steps", ax=ax2)
    ax2.set_title("Token Distribution by UT Steps", fontsize=14, fontweight="bold")
    ax2.set_xlabel("UT Steps", fontsize=12)
    ax2.set_ylabel("Generated Tokens", fontsize=12)
    plt.suptitle("")

    plt.tight_layout()
    plt.show()


def plot_error_analysis(df):
    """Analyze error patterns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Error rate by difficulty
    error_data = (
        df.groupby(["task_type", "difficulty"])
        .agg(
            {
                "is_correct_int": lambda x: 1 - x.mean()  # Error rate
            }
        )
        .reset_index()
    )

    pivot = error_data.pivot(
        index="task_type", columns="difficulty", values="is_correct_int"
    )
    pivot.plot(kind="bar", ax=ax1, colormap="Reds", width=0.8)
    ax1.set_title("Error Rate by Task and Difficulty", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Error Rate", fontsize=12)
    ax1.set_xlabel("Task Type", fontsize=12)
    ax1.legend(title="Difficulty", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Average tokens for correct vs incorrect
    token_comparison = (
        df.groupby(["task_type", "is_correct"])["generated_tokens"].mean().unstack()
    )
    token_comparison.plot(kind="bar", ax=ax2, color=["#FFB6C1", "#90EE90"])
    ax2.set_title("Avg Tokens: Correct vs Incorrect", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Average Tokens", fontsize=12)
    ax2.set_xlabel("Task Type", fontsize=12)
    ax2.legend(["Incorrect", "Correct"])
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    df = load_data(FILE_PATH)
    if df is None:
        return

    # Print summary statistics
    print_summary_table(
        df,
        ["task_type", "difficulty"],
        ["is_correct_int", "generated_tokens", "generation_time"],
        "Performance by Task Type and Difficulty",
    )

    if df["ut_steps"].nunique() > 1:
        print_summary_table(
            df, ["task_type", "ut_steps"], ["is_correct_int"], "Performance by UT Steps"
        )

    # Overall statistics
    print(f"\n{'=' * 60}")
    print(f"{'Overall Statistics':^60}")
    print(f"{'=' * 60}")
    print(f"Total Tests: {len(df)}")
    print(f"Overall Accuracy: {df['is_correct_int'].mean() * 100:.1f}%")
    print(f"Avg Generation Time: {df['generation_time'].mean():.2f}s")
    print(f"Avg Tokens Generated: {df['generated_tokens'].mean():.0f}")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    plot_accuracy_comparison(df)
    plot_ut_steps_analysis(df)
    plot_performance_scatter(df)
    plot_error_analysis(df)

    print("\n✓ Analysis complete! Check the plots above.")


if __name__ == "__main__":
    main()
