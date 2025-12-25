# 🧠 OuroTrace

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

## 🗂️ Modules Overview

- `config_loader.py`: Loads and parses experiment configuration files (JSON).
- `data_generator.py`: Generates and preprocesses datasets for supported reasoning tasks.
- `model.py`: Defines experiment classes and manages CoT evaluation logic.
- `evaluation_metrics.py`: Provides functions for analyzing and visualizing experiment results.
- `utils.py`: Contains helper functions for environment setup and path management.
- `runner.py`: Orchestrates batch experiment execution and result collection.

## License

Apache-2.0

## Tasks Evaluation

You can run end-to-end using one of the notebooks below to perform evaluation for this model on several tasks including:
- Simple reasoning tasks: n-ary addition, p-hop induction and i-GSM problems.
- Perplexity calculation: calculate the perplexity which measure the uncertainty of the model of predicting the next token, which has a strong connection to cross entropy loss.
- Reasoning primitives: variable assignment in code, math and equation of level 0 and 1 using 5-shot prompting to instruct the model.

## Notebooks
- [Kaggle Notebook](https://www.kaggle.com/code/dzung271828/ouro-trace)
- [Google Colab](https://colab.research.google.com/github/dzungphieuluuky/OuroTrace/blob/claude/ouro_trace.ipynb)