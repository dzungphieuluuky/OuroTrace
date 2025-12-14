from datetime import datetime
import os
from IPython import get_ipython
import glob


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
    if "google.colab" not in str(get_ipython()):
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