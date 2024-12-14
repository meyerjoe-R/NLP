import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a model for predicting review ratings.")
parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the model to use")
parser.add_argument("--data_file", type=str, 
                    default='/home/ubuntu/git/NLP/dissertation/data/benchmark_data/Video_Games.jsonl', help="Path to the JSON data file")
parser.add_argument("--cache_dir", type=str, default="./processed_data", help="Directory to cache processed data")
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_json_data(file_path, max_lines=25000):
    """Load data from a JSON file line by line, skipping malformed lines."""
    texts = []
    ratings = []
    with open(file_path, 'r') as fp:
        for i, line in tqdm(enumerate(fp), desc="Reading JSON lines"):
            if max_lines and i >= max_lines:
                break  # Stop after processing `max_lines` lines
            try:
                entry = json.loads(line.strip())
                # Combine title and text for input, use rating as the label
                texts.append(entry["title"] + " " + entry["text"])
                ratings.append(entry["rating"])
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i + 1}: {e}")
    return {"text": texts, "rating": ratings}

def prepare_dataset(data, tokenizer):
    """Convert raw data into tokenized datasets."""
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("rating", "labels")
    return tokenized_dataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train, validation, and test sets."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Perform the splits
    train_val_test = dataset.train_test_split(test_size=test_size + val_size, seed=42)
    val_test = train_val_test["test"].train_test_split(test_size=test_size, seed=42)

    # Combine splits into a DatasetDict
    return DatasetDict({
        "train": train_val_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

def load_or_process_data(data_file, tokenizer, cache_dir):
    """Load or preprocess data, saving to disk for reuse."""
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        try:
            print("Loading tokenized dataset from disk...")
            return DatasetDict.load_from_disk(cache_dir)
        except FileNotFoundError:
            print("Cached data is incomplete or corrupted. Reprocessing...")
            os.makedirs(cache_dir, exist_ok=True)  # Ensure the directory exists
    else:
        print("Processing raw data...")

    # Process the data
    raw_data = load_json_data(data_file)
    tokenized_dataset = prepare_dataset(raw_data, tokenizer)
    print("Splitting data into train, validation, and test sets...")
    splits = split_dataset(tokenized_dataset)

    # Pre-flatten indices to speed up future saves
    print("Pre-flattening indices...")
    splits["train"].flatten_indices()
    splits["validation"].flatten_indices()
    splits["test"].flatten_indices()

    print(f"Saving processed dataset to {cache_dir}...")
    splits.save_to_disk(cache_dir)
    return splits
    
def compute_metrics(eval_pred):
    """Compute Mean Absolute Error (MAE) for regression."""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mae = np.mean(np.abs(predictions - labels))
    return {"mae": mae}

def train_model(train_dataset, val_dataset, test_dataset, model_name):
    """Train the model on the dataset."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    output_dir = "/opt/dlami/nvme/results_reviews"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        max_steps=-1,
        save_total_limit=1,
        logging_dir=f"{output_dir}/logs",
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
    
    print('Training model...')
    trainer.train()
    model.save_pretrained(output_dir)

    # Save test predictions
    test_predictions = trainer.predict(test_dataset).predictions.squeeze()
    np.save(os.path.join(output_dir, "test_predictions.npy"), test_predictions)
    print(f"Test predictions saved to {output_dir}")

def main(args):
    # Load or preprocess data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset_splits = load_or_process_data(args.data_file, tokenizer, args.cache_dir)

    train_dataset = dataset_splits["train"]
    val_dataset = dataset_splits["validation"]
    test_dataset = dataset_splits["test"]

    # Train the model
    print("Training the model...")
    train_model(train_dataset, val_dataset, test_dataset, args.model_name)
    print("Training completed.")

if __name__ == "__main__":
    main(args)
