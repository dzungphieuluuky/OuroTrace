import torch
from datetime import datetime

BatchConfig = {
    # --- Model Settings ---
    'MODEL': {
        'path': "ByteDance/Ouro-1.4B-Thinking",
        'dtype': torch.float16,
        'use_4bit_quant': False,      # Set True if low VRAM
        'use_torch_compile': True     # Optimization
    },

    # --- Experiment Scope ---
    'INFERENCE_STEPS': [1],           # List of loop counts to test

    # --- Evaluation Logic ---
    'EVAL_SETTINGS': {
        'calculate_perplexity': True,
        'early_exit_threshold': -1.0, # -1 disables early exit
        'ppl_num_samples': 50,
        'ppl_max_length': 2048,
        'ppl_stride': 512,
    },

    # --- Logging ---
    'WANDB': {
        'enabled': True,
        'project': "ouro-looped-transformer",
        'run_name': f"run_{datetime.now().strftime('%Y%m%d_%H%M')}",
        'entity': None,
        'mode': 'offline',
    },

    # --- Data Generation ---
    'DATA': {
        'load_existing': False,
        'data_file_path': '',
        'n_ary': {'ops_levels': [4, 8], 'num_samples_per_level': 10},
        'p_hop': {'hop_levels': [2, 4], 'num_samples_per_level': 10},
        'igsm': {'num_samples_total': 20}
    },

    # --- Performance Optimization ---
    'OPTIMIZATION': {
        'enable_batch': True,
        'max_batch_size': 8,
        'max_new_token': 1024
    }
}

HolisticExperimentConfig = {
    **BatchConfig,
    'reasoning_primitives': {
        'num_samples': 50
    },
    'ENABLE_HEAVY_BENCHMARKS': False,
    'WANDB': {'enabled': False}
}