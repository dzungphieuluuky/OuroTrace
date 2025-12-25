# OuroTrace

OuroTrace is a framework for evaluating Chain-of-Thought (CoT) reasoning in Ouroboros (UT) models. It provides tools for structured prompting, dataset creation, experiment management, and result analysis.

## Key Features

- **Flexible Experimentation:** Load experiment configs from JSON, run single or batch experiments, and analyze results.
- **Dataset Utilities:** Generate and preprocess datasets for tasks like N-ary Addition, P-hop Induction, Symbolic i-GSM, and more.
- **Evaluation:** Tools for analyzing and visualizing experiment outcomes.
- **Environment Setup:** Utilities for path configuration and Colab integration.

## Quick Start

Import main components:

```python
from src import (
    load_config_from_json, create_test_datasets,
    OuroThinkingExperiment, OuroBatchExperiment,
    analyze_experiment_results
)
```

## Modules Overview

- `config_loader`: Load/process experiment configs.
- `data`: Dataset creation and preprocessing.
- `model`: Experiment classes for CoT evaluation.
- `evaluation`: Result analysis and visualization.
- `utils`: Environment and helper functions.
- `runner`: Batch experiment execution.

## License

Apache-2.0

## Notebooks

- [Kaggle Notebook](https://www.kaggle.com/code/dzung271828/ouro-trace)
- [Google Colab](https://colab.research.google.com/github/dzungphieuluuky/OuroTrace/blob/claude/ouro_trace.ipynb)