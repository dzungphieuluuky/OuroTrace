import os
import glob
import zipfile
import hashlib
import sys
from IPython import get_ipython
import pandas as pd
from typing import List, Dict

def save_results(
    simple_reasoning_results: List[Dict],
    perplexity_results: List[Dict],
    holistic_results: List[Dict],
    output_dir: str,
    overwrite: bool = True
) -> None:
    """Save experiment results to CSV files. Overwrites if overwrite=True."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, fname):
        if overwrite and os.path.exists(fname):
            os.remove(fname)
        pd.DataFrame(data).to_csv(fname, index=False)

    if simple_reasoning_results:
        stats_file = os.path.join(output_dir, "all.csv")
        save_csv(simple_reasoning_results, stats_file)
        print(f"✅ Periodic save: all results to {stats_file}")

    if perplexity_results:
        ppl_file = os.path.join(output_dir, "perplexity.csv")
        save_csv(perplexity_results, ppl_file)
        print(f"✅ Periodic save: perplexity results to {ppl_file}")

    if holistic_results:
        holistic_file = os.path.join(output_dir, "holistic.csv")
        save_csv(holistic_results, holistic_file)
        print(f"✅ Periodic save: holistic results to {holistic_file}")

def save_simple_reasoning_results(
    simple_reasoning_results: List[Dict],
    output_dir: str,
    file_name : str = "all.csv",
    overwrite: bool = True
) -> None:
    """Save only simple_reasoning_results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    if not file_name.endswith(".csv"):
        file_name += ".csv"
    stats_file = os.path.join(output_dir, file_name)
    if overwrite and os.path.exists(stats_file):
        os.remove(stats_file)
    pd.DataFrame(simple_reasoning_results).to_csv(stats_file, index=False)
    print(f"✅ Saved all results to {stats_file}")


def save_perplexity_results(
    perplexity_results: List[Dict],
    output_dir: str,
    overwrite: bool = True
) -> None:
    """Save only perplexity_results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    ppl_file = os.path.join(output_dir, "perplexity.csv")
    if overwrite and os.path.exists(ppl_file):
        os.remove(ppl_file)
    pd.DataFrame(perplexity_results).to_csv(ppl_file, index=False)
    print(f"✅ Saved perplexity results to {ppl_file}")

def save_holistic_results(
    holistic_results: List[Dict],
    output_dir: str,
    overwrite: bool = True
) -> None:
    """Save only holistic_results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    holistic_file = os.path.join(output_dir, "holistic.csv")
    if overwrite and os.path.exists(holistic_file):
        os.remove(holistic_file)
    pd.DataFrame(holistic_results).to_csv(holistic_file, index=False)
    print(f"✅ Saved holistic results to {holistic_file}")

import json

def save_config(
    config: dict,
    output_dir: str = "./default_config",
    experiment=None
) -> None:
    """Save experiment configuration and task templates to JSON files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    templates_path = os.path.join(output_dir, "task_templates.json")

    def sanitize_config(cfg):
        clean = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                clean[k] = sanitize_config(v)
            elif str(type(v)).find('torch.') != -1:
                clean[k] = str(v)
            else:
                clean[k] = v
        return clean

    # Save config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(
            sanitize_config(config),
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"✅ Configuration saved to {config_path}")

    # Save task templates if experiment is provided and has task_templates
    if experiment is not None and hasattr(experiment, "task_templates"):
        def sanitize_templates(templates):
            clean = {}
            for k, v in templates.items():
                clean[k] = {}
                for subk, subv in v.items():
                    if isinstance(subv, (str, list, dict, int, float, bool, type(None))):
                        clean[k][subk] = subv
                    else:
                        clean[k][subk] = str(subv)
            return clean

        templates_to_save = sanitize_templates(experiment.task_templates)

        with open(templates_path, 'w', encoding='utf-8') as f:
            json.dump(
                templates_to_save,
                f,
                indent=2,
                ensure_ascii=False
            )
        print(f"✅ Task templates saved to {templates_path}")

def configure_environment_paths():
    """Detect environment and configure paths"""
    try:
        if "google.colab" in str(get_ipython()):
            print("✅ Environment: Google Colab")
            base_data_path = "/content/"
            base_output_path = "/content/output/"
            environment_name = "colab"
        elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            print("✅ Environment: Kaggle")
            base_data_path = "/kaggle/input/"
            base_output_path = "/kaggle/working/"
            environment_name = "kaggle"
        else:
            print("⚠️ Environment: Local/Unknown")
            base_data_path = "./data/"
            base_output_path = "./output/"
            environment_name = "local"
    except NameError:
        print("⚠️ Non-interactive session. Using local paths.")
        base_data_path = "./data/"
        base_output_path = "./output/"
        environment_name = "local"

    os.makedirs(base_output_path, exist_ok=True)
    print(f"📂 Data Path: {base_data_path}")
    print(f"📦 Output Path: {base_output_path}")

    return base_data_path, base_output_path, environment_name

def auto_unzip_colab_content(target_dir="/content/", zip_extension="*.zip"):
    """Auto-extract zip files in Colab environment"""
    try:
        if "google.colab" not in str(get_ipython()):
            return
    except NameError:
        return

    print(f"🔎 Scanning for {zip_extension} files...")
    zip_files = glob.glob(os.path.join(target_dir, zip_extension))

    for zip_path in zip_files:
        file_name = os.path.basename(zip_path)
        base_name = os.path.splitext(file_name)[0]
        expected_output = os.path.join(target_dir, base_name)

        if os.path.exists(expected_output) and os.listdir(expected_output):
            print(f"➡️ Skipping '{file_name}' (already extracted)")
            continue

        try:
            print(f"📂 Extracting: {file_name}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
        except Exception as e:
            print(f"❌ Error: {e}")

def generate_test_id(task_type: str, difficulty: str, prompt: str) -> str:
    """Generate unique test ID"""
    unique_str = f"{task_type}_{difficulty}_{prompt}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:8]