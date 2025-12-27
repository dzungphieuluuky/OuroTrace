import os
import glob
import zipfile
import hashlib
import sys
import pandas as pd
from typing import List, Dict


import os
import pandas as pd

def concat_csv_files(filename, folder1='results_UT_4', folder2='results_UT_8', output_folder='results_concat'):
    """
    Concatenates rows from two CSV files with the same name in two folders.
    Saves the result in output_folder with the same filename.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    file1 = os.path.join(folder1, filename)
    file2 = os.path.join(folder2, filename)
    output_file = os.path.join(output_folder, filename)
    
    # Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Concatenate
    df_concat = pd.concat([df1, df2], ignore_index=True)
    
    # Save result
    df_concat.to_csv(output_file, index=False)
    print(f"Concatenated file saved to {output_file}")

def save_results(
    simple_reasoning_results: List[Dict] = None,
    perplexity_results: List[Dict] = None,
    reasoning_primitives_results: List[Dict] = None,
    benchmark_results: List[Dict] = None,
    output_dir: str = "./results",
    overwrite: bool = True,
) -> None:
    """Save experiment results to CSV files. Overwrites if overwrite=True."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, fname):
        if overwrite and os.path.exists(fname):
            os.remove(fname)
        pd.DataFrame(data).to_csv(fname, index=False)

    if simple_reasoning_results:
        stats_file = os.path.join(output_dir, "simple_reasoning.csv")
        save_csv(simple_reasoning_results, stats_file)
        print(f"✅ Periodic save: simple reasoning results to {stats_file}")

    if perplexity_results:
        ppl_file = os.path.join(output_dir, "perplexity.csv")
        save_csv(perplexity_results, ppl_file)
        print(f"✅ Periodic save: perplexity results to {ppl_file}")

    if reasoning_primitives_results:
        reasoning_primitives_file = os.path.join(output_dir, "reasoning_primitives.csv")
        save_csv(reasoning_primitives_results, reasoning_primitives_file)
        print(
            f"✅ Periodic save: reasoning primitives results to {reasoning_primitives_file}"
        )
    
    if benchmark_results:
        benchmark_file = os.path.join(output_dir, "heavy_benchmarks.csv")
        save_csv(benchmark_results, benchmark_file)
        print(f"✅ Periodic save: heavy benchmark results to {benchmark_file}")

def save_simple_reasoning_results(
    simple_reasoning_results: List[Dict],
    output_dir: str,
    file_name: str = "simple_reasoning.csv",
    overwrite: bool = True,
) -> None:
    """Save only simple_reasoning_results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    if not file_name.endswith(".csv"):
        file_name += ".csv"
    stats_file = os.path.join(output_dir, file_name)
    if overwrite and os.path.exists(stats_file):
        os.remove(stats_file)
    pd.DataFrame(simple_reasoning_results).to_csv(stats_file, index=False)
    print(f"✅ Saved simple reasoning results to {stats_file}")


def save_perplexity_results(
    perplexity_results: List[Dict], output_dir: str, overwrite: bool = True
) -> None:
    """Save only perplexity_results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    ppl_file = os.path.join(output_dir, "perplexity.csv")
    if overwrite and os.path.exists(ppl_file):
        os.remove(ppl_file)
    pd.DataFrame(perplexity_results).to_csv(ppl_file, index=False)
    print(f"✅ Saved perplexity results to {ppl_file}")


def save_reasoning_primitives_results(
    reasoning_primitives_results: List[Dict], output_dir: str, overwrite: bool = True
) -> None:
    """Save only reasoning_primitives_results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    reasoning_primitives_file = os.path.join(output_dir, "reasoning_primitives.csv")
    if overwrite and os.path.exists(reasoning_primitives_file):
        os.remove(reasoning_primitives_file)
    pd.DataFrame(reasoning_primitives_results).to_csv(
        reasoning_primitives_file, index=False
    )
    print(f"✅ Saved reasoning primitives results to {reasoning_primitives_file}")


def save_heavy_benchmark_results(
    heavy_benchmark_results: List[Dict], output_dir: str, overwrite: bool = True
) -> None:
    """Save only heavy_benchmark_results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    heavy_benchmark_file = os.path.join(output_dir, "heavy_benchmarks.csv")
    if overwrite and os.path.exists(heavy_benchmark_file):
        os.remove(heavy_benchmark_file)
    pd.DataFrame(heavy_benchmark_results).to_csv(heavy_benchmark_file, index=False)
    print(f"✅ Saved heavy benchmark results to {heavy_benchmark_file}")


import json


def save_config(
    config: dict, output_dir: str = "./default_config", experiment=None
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
            elif str(type(v)).find("torch.") != -1:
                clean[k] = str(v)
            else:
                clean[k] = v
        return clean

    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_config(config), f, indent=2, ensure_ascii=False)
    print(f"✅ Configuration saved to {config_path}")

    # Save task templates if experiment is provided and has task_templates
    if experiment is not None and hasattr(experiment, "task_templates"):

        def sanitize_templates(templates):
            clean = {}
            for k, v in templates.items():
                clean[k] = {}
                for subk, subv in v.items():
                    # Store system prompt as list of lines (one sentence per line)
                    if subk == "system" and isinstance(subv, str):
                        # Split on \n, remove empty lines, strip whitespace
                        clean[k][subk] = [
                            line.strip() for line in subv.split("\n") if line.strip()
                        ]
                    elif isinstance(
                        subv, (str, list, dict, int, float, bool, type(None))
                    ):
                        clean[k][subk] = subv
                    else:
                        clean[k][subk] = str(subv)
            return clean

        templates_to_save = sanitize_templates(experiment.task_templates)

        with open(templates_path, "w", encoding="utf-8") as f:
            json.dump(templates_to_save, f, indent=2, ensure_ascii=False)
        print(f"✅ Task templates saved to {templates_path}")


def configure_environment_paths():
    """Detect environment and configure paths"""
    try:
        from IPython import get_ipython
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
        from IPython import get_ipython
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

if __name__ == "__main__":
    # Example usage
    import argparse
    # Input output folder, filename, folder 1 and folder 2
    parser = argparse.ArgumentParser(description="Concatenate CSV files from two folders.")
    parser.add_argument(
        "--filename", type=str, required=True, help="Name of the CSV file to concatenate"
    )
    parser.add_argument(
        "--folder1", type=str, default="results_UT_4", help="First folder containing the CSV file"
    )
    parser.add_argument(
        "--folder2", type=str, default="results_UT_8", help="Second folder containing the CSV file"
    )
    parser.add_argument(
        "--output_folder", type=str, default="results_concat", help="Output folder for the concatenated CSV file"
    )
    args = parser.parse_args()
    concat_csv_files(args.filename, args.folder1, args.folder2, args.output_folder)