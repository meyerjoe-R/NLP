import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, load_metric

os.environ["WANDB_DISABLED"] = "true"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a model for the Semantic Textual Similarity Benchmark.")
parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the model to use")
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def prepare_sts_datasets(dataset, tokenizer):
    """Tokenize paired sentences for the STS task."""
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("score", "labels")
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2"])
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute Mean Absolute Error (MAE) for STS."""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mae_metric = load_metric("mae")
    return mae_metric.compute(predictions=predictions, references=labels)

def train_sts_model(train_dataset, val_dataset, test_dataset, model_name):
    """Train a model for STS Benchmark."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    output_dir = "./transformer_results_sts"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=f"{output_dir}/logs",
        # max_steps = 5,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,  # Lower MAE is better
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    model.save_pretrained(output_dir)

    # Generate and save predictions on the test set
    test_predictions = trainer.predict(test_dataset).predictions.squeeze()
    np.save(os.path.join(output_dir, "test_predictions.npy"), test_predictions)
    print(f"Test predictions saved to {output_dir}")

    return model

def main(args):
    # Load the STS Benchmark dataset from Hugging Face
    dataset = load_dataset("sentence-transformers/stsb")

    # Load tokenizer and prepare datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = prepare_sts_datasets(dataset, tokenizer)

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # Train the model
    print("Training model for STS Benchmark...")
    train_sts_model(train_dataset, val_dataset, test_dataset, args.model_name)
    print("Training completed.")

if __name__ == "__main__":
    main(args)
