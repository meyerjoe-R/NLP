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

os.environ["WANDB_DISABLED"] = "true"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a model for personality trait prediction.")
parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the BERT model to use")
parser.add_argument("--use_elmo", action="store_true", help="Use ELMo instead of BERT")
parser.add_argument("--text_column", type=str, required=True, help="Name of the text column in the dataset")
parser.add_argument("--target_columns", type=str, nargs="+", default=["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"],
                    help="List of target column(s) in the dataset")

args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        print(f'output: {output}')
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

def train_elmo_model(elmo, X_train_ids, target_train, target_name, num_epochs = 5, 
                     debug = True, max_steps = 1e11):
    
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

def prepare_datasets(train_df, val_df, test_df, tokenizer, text_column):
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_function(example):
        return tokenizer(example[text_column], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    return train_dataset, val_dataset, test_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mae_metric = load_metric("mae")
    return {"mae": mae_metric.compute(predictions=predictions, references=labels)["mae"]}

def train_bert_model(target, train_dataset, val_dataset, test_dataset, model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    output_dir = f"./results_{target}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        # max_steps = 5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        fp16=True
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
    np.save(os.path.join(output_dir, f"{target}_test_predictions.npy"), test_predictions)
    print(f"Test predictions for {target} saved to {output_dir}")
    
    return model

def main(args):
    root = '/home/ubuntu/git/NLP/dissertation/data/'

    # Load CSV files
    train_df = pd.read_csv(f'{root}/benchmark_data/Personality Datasets - Reddit_train_set.csv')
    val_df = pd.read_csv(f'{root}/benchmark_data/Personality Datasets - Reddit_val_set.csv')
    test_df = pd.read_csv(f'{root}/benchmark_data/Personality Datasets - Reddit_eval_set.csv')
    
    # Add text cleaning
    train_df[args.text_column] = train_df[args.text_column].apply(lambda text: text if isinstance(text, str) else "")
    test_df[args.text_column] = test_df[args.text_column].apply(lambda text: text.strip() if isinstance(text, str) and text.strip() else "")
    val_df[args.text_column] = val_df[args.text_column].apply(lambda text: text.strip() if isinstance(text, str) and text.strip() else "")

    if args.use_elmo:
        # Initialize ELMo
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        print('Loading ELMo...')
        # Initialize ELMo with local caching
        print("Initializing ELMo...")
        elmo = initialize_elmo(options_file, weight_file)
        
        # Prepare data for ELMo
        X_train_ids = prepare_elmo_data(train_df, args.text_column)
        X_test_ids = prepare_elmo_data(test_df, args.text_column)  # Prepare test IDs
        
        if torch.isnan(X_train_ids).any() or torch.isinf(X_train_ids).any():
            print("Input data contains NaN or Inf!")
            return

        for target in args.target_columns:
            print(f"Training ELMo model for target: {target}")
            target_train = torch.tensor(train_df[target].values, dtype=torch.float32).to(device)
            print(f'target train shape: {target_train}')
            model = load_or_train_elmo_model(elmo, X_train_ids, target_train, target)
            
            # Predict on the test set
            print(f"Generating predictions for target: {target}")
            predict_elmo_model(model, X_test_ids, target)
    else:
        # Prepare tokenizer and datasets for BERT-based approach
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df, tokenizer, args.text_column)

        for target in args.target_columns:
            print(f"Training BERT model for target: {target}")
            model = train_bert_model(
                target,
                train_dataset.map(lambda x: {"labels": x[target]}, remove_columns=args.target_columns),
                val_dataset.map(lambda x: {"labels": x[target]}, remove_columns=args.target_columns),
                test_dataset,
                args.model_name
            )
            print(f"Training completed for target: {target}")

if __name__ == "__main__":
    main(args)