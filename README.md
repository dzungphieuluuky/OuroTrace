# OuroTrace

OuroTrace is a framework for evaluating the Chain-of-Thought (CoT) reasoning capabilities of Ouroboros (UT) models. It leverages tightly controlled few-shot prompting, guardrails, and step prefixes to enforce strict, step-by-step logic and maximize accuracy on structured reasoning tasks.

## Features

- **Configurable Experiments:** Easily load and post-process experiment configurations from JSON files.
- **Data Utilities:** Create and preprocess datasets for tasks such as N-ary Addition, P-hop Induction, Symbolic i-GSM, and more.
- **Model Experiments:** Run and analyze both single and batch experiments using the OuroThinkingExperiment and OuroBatchExperiment classes.
- **Evaluation Tools:** Analyze experiment results and perform holistic evaluations.
- **Environment Setup:** Utilities for configuring environment paths and handling Colab content.

## Usage

Import core functions and classes directly from the package:

```python
from src import (
    load_config_from_json, create_test_datasets, OuroThinkingExperiment,
    run_batch_experiment, analyze_experiment_results
)
```

## Modules

- `config_loader`: Load and process experiment configurations.
- `utils`: Environment setup and utility functions.
- `data`: Dataset creation and preprocessing.
- `model`: Experiment classes for running CoT evaluations.
- `evaluation`: Analysis and result processing.
- `runner`: Batch and holistic experiment runners.

## License

 Apache-2.0 license

 ## Run This Notebook

- [Open in Kaggle](https://www.kaggle.com/code/dzung271828/ouro-trace)
- [Open in Google Colab](https://colab.research.google.com/github/dzungphieuluuky/OuroTrace/blob/claude/ouro_trace.ipynb)