# This script needs these libraries to be installed:
#   numpy, transformers, datasets
"""This is a simple example of using Weights & Biases (wandb) to log training metrics"""

import wandb

import os
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


# download prepare the data
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(300))

small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
small_eval_dataset = small_train_dataset.map(tokenize_function, batched=True)

# download the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5
)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "my-awesome-project"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

# pass "wandb" to the 'report_to' parameter to turn on wandb logging
training_args = TrainingArguments(
    output_dir="models",
    report_to="wandb",
    logging_steps=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=20,
    max_steps=100,
    save_steps=100,
)

# define the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
