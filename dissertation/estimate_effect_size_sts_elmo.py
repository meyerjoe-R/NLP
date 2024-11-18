import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from allennlp.modules.elmo import Elmo, batch_to_ids
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a model for the STS Benchmark using ELMo embeddings.")
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def initialize_elmo():
    """Initialize the ELMo model."""
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0.5).to(device)
    return elmo

class ElmoRegressionModel(nn.Module):
    """Regression model using ELMo embeddings."""
    def __init__(self, input_dim=2048, dropout_rate=0.5):
        super(ElmoRegressionModel, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, elmo_avg1, elmo_avg2):
        # Concatenate averaged embeddings of both sentences
        combined = torch.cat([elmo_avg1, elmo_avg2], dim=1)
        combined = self.dropout(combined)
        return self.fc(combined).squeeze()

def prepare_elmo_inputs(sentences):
    """Convert sentences to ELMo-compatible character IDs."""
    tokenized_sentences = [sentence.split() for sentence in sentences]
    return batch_to_ids(tokenized_sentences).to(device)

def compute_elmo_embeddings(elmo, sentence_ids):
    """Compute the average ELMo embeddings for a batch of sentences."""
    elmo_output = elmo(sentence_ids)["elmo_representations"][0]
    return torch.mean(elmo_output, dim=1)

def train_elmo_model(model, elmo, dataloader, num_epochs=5, lr=0.001):
    """Train the ELMo regression model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sent1_ids, sent2_ids, targets = batch
            sent1_ids, sent2_ids, targets = sent1_ids.to(device), sent2_ids.to(device), targets.to(device)

            # Compute ELMo embeddings
            elmo_avg1 = compute_elmo_embeddings(elmo, sent1_ids)
            elmo_avg2 = compute_elmo_embeddings(elmo, sent2_ids)

            # Forward pass
            outputs = model(elmo_avg1, elmo_avg2)
            loss = criterion(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

def predict_elmo_model(model, elmo, dataloader, output_dir="./elmo_results"):
    """Generate and save predictions for the test set using the trained ELMo model."""
    model.eval()
    predictions = []

    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, "test_predictions.npy")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            sent1_ids, sent2_ids = batch
            sent1_ids, sent2_ids = sent1_ids.to(device), sent2_ids.to(device)

            # Compute ELMo embeddings
            elmo_avg1 = compute_elmo_embeddings(elmo, sent1_ids)
            elmo_avg2 = compute_elmo_embeddings(elmo, sent2_ids)

            # Forward pass
            outputs = model(elmo_avg1, elmo_avg2)
            predictions.extend(outputs.cpu().numpy())

    # Save predictions to file
    np.save(predictions_file, predictions)
    print(f"Test predictions saved to {predictions_file}")

def main():
    # Load the STS Benchmark dataset
    dataset = load_dataset("sentence-transformers/stsb")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    # Prepare ELMo inputs and targets
    elmo = initialize_elmo()
    train_sent1_ids = prepare_elmo_inputs(train_data["sentence1"])
    train_sent2_ids = prepare_elmo_inputs(train_data["sentence2"])
    train_targets = torch.tensor(train_data["score"], dtype=torch.float32).to(device)

    test_sent1_ids = prepare_elmo_inputs(test_data["sentence1"])
    test_sent2_ids = prepare_elmo_inputs(test_data["sentence2"])

    # Create DataLoader
    train_dataset = TensorDataset(train_sent1_ids, train_sent2_ids, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = TensorDataset(test_sent1_ids, test_sent2_ids)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize regression model
    model = ElmoRegressionModel().to(device)

    # Train the model
    print("Training ELMo regression model...")
    train_elmo_model(model, elmo, train_loader)

    # Save test predictions
    print("Saving test predictions...")
    predict_elmo_model(model, elmo, test_loader)

if __name__ == "__main__":
    main()
