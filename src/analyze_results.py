import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

FILE_PATH = "ouro_acc_20251215_061001.csv"

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Convert boolean is_correct to integer (0 or 1) for averaging
        df['is_correct_int'] = df['is_correct'].astype(int)
        return df
    except FileNotFoundError:
        print(f"ERROR: Could not find '{filepath}'. Please ensure the file exists.")
        return None

def draw_text_table(df, group_cols, value_cols, title):
    """Helper to print formatted ASCII tables"""
    print(f"\n{'='*20} {title} {'='*20}")
    
    # Group and aggregate
    agg_dict = {col: 'mean' for col in value_cols}
    # Add count to the first column metric to see sample size
    agg_dict[value_cols[0]] = ['mean', 'count']
    
    summary = df.groupby(group_cols).agg(agg_dict)
    
    # Flatten multi-level columns for display
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Rename for readability
    rename_map = {
        'is_correct_int_mean': 'Accuracy',
        'is_correct_int_count': 'Samples',
        'generated_tokens_mean': 'Avg Tokens',
        'generation_time_mean': 'Avg Time(s)'
    }
    summary = summary.rename(columns=rename_map)
    
    # Format percentages
    summary['Accuracy'] = (summary['Accuracy'] * 100).apply(lambda x: f"{x:.2f}%")
    summary['Avg Tokens'] = summary['Avg Tokens'].apply(lambda x: f"{x:.1f}")
    summary['Avg Time(s)'] = summary['Avg Time(s'].apply(lambda x: f"{x:.2f}")
    
    print(summary)
    return summary

def plot_accuracy_by_config(df):
    """Bar chart comparing Accuracy across Task Type and Difficulty"""
    plt.figure(figsize=(14, 7))
    
    # Create a composite column for clearer x-axis labeling if needed, 
    # or just use hue. Here we use Task Type as X and Difficulty as Hue.
    ax = sns.barplot(
        data=df,
        x='task_type',
        y='is_correct_int',
        hue='difficulty',
        palette='viridis',
        errorbar=None  # Remove error bars for cleaner comparison of means
    )
    
    plt.title('Accuracy by Task Type and Difficulty', fontsize=16)
    plt.ylabel('Accuracy (0-1)', fontsize=12)
    plt.xlabel('Task Type', fontsize=12)
    plt.legend(title='Difficulty', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    plt.show()

def plot_resource_usage(df):
    """Scatter plot to analyze Time vs Tokens, colored by Correctness"""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    sns.scatterplot(
        data=df,
        x='generated_tokens',
        y='generation_time',
        hue='is_correct',
        style='task_type', # Different shapes for different tasks
        palette={True: 'green', False: 'red'},
        s=100, # Marker size
        alpha=0.7
    )
    
    plt.title('Performance Analysis: Generation Time vs Tokens Generated', fontsize=16)
    plt.xlabel('Generated Tokens', fontsize=12)
    plt.ylabel('Generation Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Is Correct / Task', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_ut_steps_impact(df):
    """Line chart to see how UT Steps affects accuracy"""
    # Only plot if there is variation in ut_steps
    if df['ut_steps'].nunique() <= 1:
        print("\n[INFO] Skipping UT Steps graph: Only one unique step value found.")
        return

    plt.figure(figsize=(12, 6))
    
    sns.lineplot(
        data=df,
        x='ut_steps',
        y='is_correct_int',
        hue='task_type',
        marker='o',
        linewidth=2.5
    )
    
    plt.title('Impact of UT Steps on Accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('UT Steps', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Task Type')
    
    plt.tight_layout()
    plt.show()

def plot_token_efficiency_box(df):
    """Box plot to see the spread of tokens generated for Correct vs Incorrect"""
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(
        data=df,
        x='task_type',
        y='generated_tokens',
        hue='is_correct',
        palette={True: '#90EE90', False: '#FFB6C1'} # Light green and light red
    )
    
    plt.title('Token Usage Distribution: Correct vs Incorrect', fontsize=16)
    plt.ylabel('Generated Tokens', fontsize=12)
    plt.xlabel('Task Type', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    df = load_data(FILE_PATH)
    if df is not None:
        # 1. Draw Console Tables for precise numbers
        draw_text_table(df, 
                       group_cols=['task_type', 'difficulty'], 
                       value_cols=['is_correct_int', 'generated_tokens', 'generation_time'],
                       title="Breakdown by Task & Difficulty")
        
        # Check if UT Steps vary, if so, make a table for it
        if df['ut_steps'].nunique() > 1:
            draw_text_table(df,
                           group_cols=['task_type', 'ut_steps'],
                           value_cols=['is_correct_int'],
                           title="Breakdown by UT Steps")

        print("\nGenerating visual plots...")
        
        # 2. Draw Visual Graphs
        plot_accuracy_by_config(df)
        plot_ut_steps_impact(df)
        plot_resource_usage(df)
        plot_token_efficiency_box(df)
        
        print("Done. Check the open windows for graphs.")

if __name__ == "__main__":
    main()