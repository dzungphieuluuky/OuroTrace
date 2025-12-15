import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.config_loader import load_config_from_json, post_process_config
from src.utils import configure_environment_paths, auto_unzip_colab_content
from src.runner import run_batch_experiment
from src.evaluation import analyze_experiment_results

# 1. Setup Environment
DATA_PATH, OUTPUT_PATH, ENV = configure_environment_paths()
auto_unzip_colab_content(DATA_PATH)

# 2. Load Configuration
if __name__ == "__main__":
    # 1. Load Base Config (BatchConfig)
    BatchConfig = load_config_from_json('configs/batch_config.json')

    # 2. Load Holistic Extension
    HolisticExtension = load_config_from_json('configs/holistic_extension.json')
    
    if BatchConfig and HolisticExtension:
        # 3. Merge configs to create HolisticExperimentConfig
        HolisticExperimentConfig = BatchConfig.copy()
        
        # Deep merge for WANDB and append other keys
        for key, value in HolisticExtension.items():
            if key in HolisticExperimentConfig and isinstance(value, dict) and isinstance(HolisticExperimentConfig[key], dict):
                HolisticExperimentConfig[key].update(value)
            else:
                HolisticExperimentConfig[key] = value

        # 4. Post-process configurations
        HolisticExperimentConfig = post_process_config(HolisticExperimentConfig)

        print("✅ Configuration loaded")
        print("\n--- HolisticExperimentConfig Processed ---")
        print(f"Model Path: {HolisticExperimentConfig['MODEL']['path']}")
        print(f"WANDB Run: {HolisticExperimentConfig['WANDB']['run_name']}")
    else:
        print("❌ Failed to load configuration files. Exiting.")
        exit(1)

    # 3. Experiment Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🕒 Timestamp: {timestamp}")
    print("\n" + "=" * 50 + "\n🚀 STARTING EXPERIMENT\n" + "=" * 50)

    # 4. Run
    acc_results, ppl_results, holistic_results = run_batch_experiment(HolisticExperimentConfig)

    # 5. Save Results
    df_acc = pd.DataFrame(acc_results)
    df_ppl = pd.DataFrame(ppl_results)
    df_hollistic = pd.DataFrame(holistic_results)

    RUN_RESULTS_NAME = f"run_{timestamp}"
    os.makedirs(os.path.join(OUTPUT_PATH, RUN_RESULTS_NAME), exist_ok=True)
    acc_path = os.path.join(OUTPUT_PATH, RUN_RESULTS_NAME, f"ouro_acc_{timestamp}.csv")
    ppl_path = os.path.join(OUTPUT_PATH, RUN_RESULTS_NAME, f"ouro_ppl_{timestamp}.csv")
    hol_path = os.path.join(OUTPUT_PATH, RUN_RESULTS_NAME, f"ouro_holistic_{timestamp}.csv")
    cfg_path = os.path.join(OUTPUT_PATH, RUN_RESULTS_NAME, f"ouro_config_{timestamp}.yaml")

    df_acc.to_csv(acc_path, index=False)
    if not df_ppl.empty:
        df_ppl.to_csv(ppl_path, index=False)
    if not df_hollistic.empty:
        df_hollistic.to_csv(hol_path, index=False)

    # Helper to sanitize config for YAML
    def sanitize_config_yaml(cfg):
        clean = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                clean[k] = sanitize_config_yaml(v)
            elif str(type(v)).find("torch.") != -1:
                clean[k] = str(v)
            else:
                clean[k] = v
        return clean

    with open(cfg_path, "w") as f:
        yaml.dump(sanitize_config_yaml(HolisticExperimentConfig), f)

    print(f"\n💾 Results saved to {os.path.join(OUTPUT_PATH, RUN_RESULTS_NAME)}")

    # 6. Visualization & Reporting
    if not df_acc.empty:
        print("\n" + "=" * 50 + "\n📊 VISUALIZATION\n" + "=" * 50)

        # Summary Tables
        summary = analyze_experiment_results(acc_results)
        print("\n--- Summary Statistics ---")
        print(summary)

        # Plotting
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Plot 1: Accuracy
            if 'ut_steps' in df_acc.columns and 'task_type' in df_acc.columns:
                acc_summary = (
                    df_acc.groupby(["task_type", "ut_steps"])["is_correct"].mean().reset_index()
                )
                sns.barplot(
                    data=acc_summary, x="ut_steps", y="is_correct", hue="task_type", ax=axes[0]
                )
                axes[0].set_title("Accuracy by UT Steps")
                axes[0].set_ylabel("Accuracy")
                axes[0].yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: "{:.0%}".format(y))
                )

                # Plot 2: Time
                time_summary = (
                    df_acc.groupby(["task_type", "ut_steps"])["generation_time"]
                    .mean()
                    .reset_index()
                )
                sns.barplot(
                    data=time_summary,
                    x="ut_steps",
                    y="generation_time",
                    hue="task_type",
                    ax=axes[1],
                )
                axes[1].set_title("Inference Time (s) by UT Steps")

                # Plot 3: Token Count
                if 'generated_tokens' in df_acc.columns:
                    sns.boxplot(
                        data=df_acc, x="ut_steps", y="generated_tokens", hue="task_type", ax=axes[2]
                    )
                    axes[2].set_title("Generated Tokens Distribution")

                plt.tight_layout()
                plt.show()
                # Save plot
                plt.savefig(os.path.join(OUTPUT_PATH, RUN_RESULTS_NAME, f"results_plot_{timestamp}.png"))
                print(f"📊 Plot saved to results_plot_{timestamp}.png")

        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    else:
        print("⚠️ No results to visualize.")

    print("\n🏁 Experiment Complete.\n")
    if not df_acc.empty:
        print("Top 20 Accuracy Report:\n")
        print(df_acc.head(20))
    if not df_ppl.empty:
        print("Perplexity Report:\n")
        print(df_ppl.head(20))