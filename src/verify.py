# test_fixes.py
from model import OuroBatchExperiment
import torch
# Test initialization
experiment = OuroBatchExperiment(
    model_path="test/path",
    dtype=torch.float16,
    use_torch_compile=False,
    max_batch_size=4
)

# Test that all attributes exist
assert hasattr(experiment, '_template_cache')
assert hasattr(experiment, '_model_cache')
assert hasattr(experiment, '_templates_precomputed')
assert hasattr(experiment, '_cleanup_between_batches')
assert hasattr(experiment, '_optimize_model_for_ut_steps')
assert hasattr(experiment, '_get_cached_template')

print("✅ All attributes properly initialized")