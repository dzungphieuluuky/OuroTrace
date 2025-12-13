import json
import torch
from datetime import datetime
from typing import Dict, Any

def load_config_from_json(filepath: str) -> Dict[str, Any]:
    """Tải cấu hình từ file JSON vào Python dictionary."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file cấu hình tại {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Lỗi khi giải mã JSON trong {filepath}: {e}")
        return {}

def post_process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Xử lý các chuỗi placeholder thành các đối tượng Python thực tế.
    """
    # Xử lý kiểu dữ liệu Torch
    if config.get('MODEL', {}).get('dtype') == "torch.float16":
        config['MODEL']['dtype'] = torch.float16
    elif config.get('MODEL', {}).get('dtype') == "torch.float32":
        config['MODEL']['dtype'] = torch.float32
    elif config.get('MODEL', {}).get('dtype') == "torch.bfloat16":
        config['MODEL']['dtype'] = torch.bfloat16

    # Xử lý timestamp tự động cho WANDB
    if config.get('WANDB', {}).get('run_name') == "auto_timestamp":
        config['WANDB']['run_name'] = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
    return config