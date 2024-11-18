import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset, load_metric
from allennlp.modules.elmo import Elmo, batch_to_ids
from scipy.stats import pearsonr
from tqdm import tqdm
import json
from datasets import Dataset, DatasetDict

os.environ["WANDB_DISABLED"] = "true"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a model for predicting review ratings.")
parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the model to use")
parser.add_argument("--cache_dir", type=str, default="./processed_data", help="Directory to cache processed data")
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_processed_data(cache_dir):
    """Load train, validation, and test datasets from processed folders."""
    dataset_splits = {}
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(cache_dir, split)
        if os.path.exists(split_path) and os.path.isdir(split_path):
            print(f"Loading {split} data from {split_path}...")
            dataset_splits[split] = Dataset.load_from_disk(split_path)
            print(dataset_splits)
        else:
            raise FileNotFoundError(f"{split.capitalize()} folder not found in {cache_dir}. Ensure data is processed correctly.")
    return DatasetDict(dataset_splits)


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


def initialize_elmo(options_file_url, weight_file_url, local_dir="./elmo_weights"):
    """Initialize ELMo model with local caching for weights and options."""
    os.makedirs(local_dir, exist_ok=True)

    # Local file paths
    options_file = os.path.join(local_dir, "elmo_options.json")
    weight_file = os.path.join(local_dir, "elmo_weights.hdf5")

    # Download files if they don't exist locally
    if not os.path.exists(options_file):
        print(f"Downloading ELMo options file to {options_file}")
        torch.hub.download_url_to_file(options_file_url, options_file)

    if not os.path.exists(weight_file):
        print(f"Downloading ELMo weights file to {weight_file}")
        torch.hub.download_url_to_file(weight_file_url, weight_file)

    # Initialize ELMo model
    elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0.5, requires_grad=True).to(device)
    return elmo


# ELMo Model Definition
class ElmoRegressionModel(nn.Module):
    def __init__(self, elmo_model, input_dim=1024, dropout_rate=0.5):
        super(ElmoRegressionModel, self).__init__()
        self.elmo = elmo_model
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, input_tensor):
        elmo_output = self.elmo(input_tensor)['elmo_representations'][0]
        mean_embedding = torch.mean(elmo_output, dim=1)
        x = self.dropout(mean_embedding)
        output = self.output_layer(x)
        return output

def prepare_elmo_data(dataframe, text_column):
    """Prepare ELMo-compatible input data."""
    sentences = dataframe[text_column].tolist()
    ids = batch_to_ids([sentence.split() for sentence in sentences]).to(device)
    return ids

def load_or_train_elmo_model(elmo, X_train_ids, target_train, target_name):
    """Load a saved ELMo model or train a new one."""
    output_dir = f"./elmo_results_{target_name}"
    model_path = os.path.join(output_dir, f"{target_name}_model.pth")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Training new model for {target_name}")
    model = train_elmo_model(elmo, X_train_ids, target_train, target_name)

    return model

def train_elmo_model(elmo, X_train_ids, target_train, target_name,
                     num_epochs = 5, debug = True, max_steps = 1e11):
    """Train the ELMo model for a single target."""
    model = ElmoRegressionModel(elmo).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader for the specific target
    train_dataset = TensorDataset(X_train_ids, target_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print(f'x train ids: {X_train_ids}')
    print(f'x train target: {target_train}')

    step_counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_inputs, batch_targets in tqdm(train_loader):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            if debug:
                print(f'batch inputs: {batch_inputs}')
                print(f'batch targets: {batch_targets}')
            
            outputs = model(batch_inputs).squeeze()
            if debug:
                print(f'prediction: {outputs}')
            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_counter +=1
            if step_counter >= max_steps:
                print('Max steps reached')
                break
        
        print(f"Epoch {epoch + 1}, Loss for {target_name}: {total_loss / len(train_loader)}")

    # Save the model for this target
    output_dir = f"./elmo_results_{target_name}"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"{target_name}_model.pth"))
    print(f"Model for {target_name} saved to {output_dir}")

    return model

def predict_elmo_model(model, X_test_ids, target_name):
    """Generate and save predictions for the test set using the trained ELMo model."""
    model.eval()
    predictions = []

    # Create DataLoader for the test set
    test_dataset = TensorDataset(X_test_ids)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        for batch_inputs in test_loader:
            batch_inputs = batch_inputs[0].to(device)  # Unpack the tuple
            outputs = model(batch_inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())

    # Save predictions to file
    output_dir = f"./elmo_results_{target_name}"
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, f"{target_name}_test_predictions.npy")
    np.save(predictions_file, predictions)
    print(f"Test predictions for {target_name} saved to {predictions_file}")
    
def convert_text_to_ids(texts, device, batch_size=512, max_seq_length=512):
    """Convert text to ELMo-compatible input IDs with a progress bar and consistent tensor sizes."""
    tokenized_texts = [text.split() for text in texts]
    ids_list = []
    
    print("Converting text to IDs...")
    for i in tqdm(range(0, len(tokenized_texts), batch_size), desc="Processing Text Batches"):
        batch = tokenized_texts[i : i + batch_size]
        ids = batch_to_ids(batch).to(device)

        # Truncate or pad to ensure consistent tensor size
        max_len = min(ids.size(1), max_seq_length)
        if ids.size(1) > max_seq_length:
            ids = ids[:, :max_len, :]
        elif ids.size(1) < max_seq_length:
            padding = torch.zeros(ids.size(0), max_seq_length - ids.size(1), ids.size(2), device=ids.device, dtype=ids.dtype)
            ids = torch.cat([ids, padding], dim=1)

        ids_list.append(ids)

    # Concatenate all IDs into a single tensor
    return torch.cat(ids_list, dim=0)


    
def main(args):
    cache_dir = args.cache_dir  # Path to the processed data directory

    # Load processed datasets
    print("Loading processed datasets...")
    dataset_splits = load_processed_data(cache_dir)
    
    train_text = dataset_splits["train"]['text']
    # val_text = dataset_splits['validation']["text"]
    test_text = dataset_splits['test']["text"]
    
    train_text = [text if isinstance(text, str) else "" for text in train_text]
    test_text = [text.strip() for text in test_text if text.strip()]
    
    
    train_labels = dataset_splits["train"]['labels']
    # val_labels = dataset_splits["validation"]['labels']
    # test_labels = dataset_splits["test"]['labels']
    
    # Initialize ELMo
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    
    print("Initializing ELMo...")
    elmo = initialize_elmo(options_file, weight_file)

    print('Creating IDs for train and test sets...')
    # Convert text to IDs
    X_train_ids = convert_text_to_ids(train_text, device)
    X_test_ids = convert_text_to_ids(test_text, device)

    if torch.isnan(X_train_ids).any() or torch.isinf(X_train_ids).any():
        print("Input data contains NaN or Inf!")
        return

    target_name = 'review' 
     
    target_train = torch.tensor(train_labels, dtype=torch.float32).to(device)

    print('Training....')
    
    # Note: Assuming y_train, y_val, y_test are processed accordingly for each target
    model = load_or_train_elmo_model(elmo, X_train_ids, target_train, target_name)

    # Predict on the test set
    test_predictions = predict_elmo_model(model, X_test_ids, target_name)
            
if __name__ == "__main__":
    main(args)