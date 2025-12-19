import os
import glob
import zipfile
import hashlib
import sys
from IPython import get_ipython
import pandas as pd
from typing import List, Dict

def save_results(
    all_results: List[Dict],
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

    if all_results:
        all_file = os.path.join(output_dir, "all_latest.csv")
        save_csv(all_results, all_file)
        print(f"✅ Periodic save: all results to {all_file}")

    if perplexity_results:
        ppl_file = os.path.join(output_dir, "perplexity_latest.csv")
        save_csv(perplexity_results, ppl_file)
        print(f"✅ Periodic save: perplexity results to {ppl_file}")

    if holistic_results:
        holistic_file = os.path.join(output_dir, "holistic_latest.csv")
        save_csv(holistic_results, holistic_file)
        print(f"✅ Periodic save: holistic results to {holistic_file}")

def save_config(
    config: dict,
    output_dir: str
) -> None:
    """Save experiment configuration to a YAML file (once at the start)."""
    import os
    import yaml

    os.makedirs(output_dir, exist_ok=True)

    # sanitize config before saving
    clean = {}
    for k, v in config.items():
        if isinstance(v, dict):
            clean[k] = {kk: str(vv) for kk, vv in v.items()}
        else:
            clean[k] = str(v)

    config_file = os.path.join(output_dir, "config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(clean, f)
    print(f"✅ Saved config to {config_file}")

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