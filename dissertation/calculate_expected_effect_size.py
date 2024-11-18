import pandas as pd
import numpy as np
import os
from datasets import Dataset, DatasetDict, load_dataset

personality_constructs = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]
tasks = ["agreeableness", "conscientiousness", "extraversion", "neuroticism", "openness", "review", "sts"]

def find_folders_with_results(path, search_term='results_'):
    results_folders = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            if search_term in dir_name:
                results_folders.append(os.path.join(root, dir_name))
    return results_folders

def load_processed_data(cache_dir = './processed_data'):
    """Load train, validation, and test datasets from processed folders."""
    dataset_splits = {}
    for split in ["test"]:
        split_path = os.path.join(cache_dir, split)
        if os.path.exists(split_path) and os.path.isdir(split_path):
            print(f"Loading {split} data from {split_path}...")
            dataset_splits[split] = Dataset.load_from_disk(split_path)
        else:
            raise FileNotFoundError(f"{split.capitalize()} folder not found in {cache_dir}. Ensure data is processed correctly.")
    return DatasetDict(dataset_splits)

def find_npy_files_in_folders(folder_list):
    npy_files = {}
    for folder in folder_list:
        folder_files = []
        for root, _, files in os.walk(folder):
            for file_name in files:
                if file_name.endswith(".npy"):
                    folder_files.append(os.path.join(root, file_name))
        if folder_files:
            npy_files[folder] = folder_files
    return npy_files

def load_npy_files(npy_file_dict):
    loaded_data = {}
    for folder, files in npy_file_dict.items():
        folder_data = {}
        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                folder_data[file_name] = np.load(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        loaded_data[folder] = folder_data
    return loaded_data

def map_task_from_filename(folder, file_name):
    if "review" in folder.lower():
        return "review"
    elif "sts" in folder.lower():
        return "sts"
    for task in tasks:
        if task in file_name.lower():
            return task
    return None

# Function to load predictions into a dictionary by task and model type
def organize_predictions(npy_data):
    task_predictions = {}
    for folder, files in npy_data.items():
        model_type = "elmo" if "elmo_" in folder else "transformer"
        for file_name, data in files.items():
            task = map_task_from_filename(folder, file_name)
            if task:
                task_predictions[(task, model_type)] = data
    return task_predictions

# Load ground truths for each task
def load_ground_truths():
    ground_truths = {}

    # Load Reviews ground truth
    reviews_dataset = load_processed_data()
    ground_truths["review"] = reviews_dataset['test']['labels']  # Adjust 'labels' if needed

    # Load STS ground truth
    sts_dataset = load_dataset("sentence-transformers/stsb", split="test")
    ground_truths["sts"] = sts_dataset["score"]  

    
    
    # Load Personality ground truth
    root = '/home/ubuntu/git/NLP/dissertation/data/'
    personality_ground_truth = pd.read_csv(f'{root}/benchmark_data/Personality Datasets - Reddit_eval_set.csv')
    print(personality_ground_truth)
    for construct in personality_constructs:
        ground_truths[construct] = personality_ground_truth[construct].values

    return ground_truths

# Function to calculate Cohen's d
def cohen_d(x, y):
    diff_mean = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
    return diff_mean / pooled_std if pooled_std != 0 else np.nan

# Finding result folders and .npy files
result_folders = find_folders_with_results('/home/ubuntu/git/NLP')
npy_files = find_npy_files_in_folders(result_folders)

# Loading .npy files into a dictionary
npy_data = load_npy_files(npy_files)

npy_data = dict(sorted(npy_data.items()))

# Print loaded data
for folder, files in npy_data.items():
    print(f"Folder: {folder}")
    for file_name, data in files.items():
        print(f"  File: {file_name}, Data shape: {data.shape if isinstance(data, np.ndarray) else 'N/A'}")

# Load Elmo and Transformer predictions into a unified dictionary
all_predictions = organize_predictions(npy_data)

# Align predictions into DataFrames by task
task_dataframes = {}
for task in tasks:
    elmo_key = (task, "elmo")
    transformer_key = (task, "transformer")
    
    # Check if both models have predictions for the task
    if elmo_key in all_predictions and transformer_key in all_predictions:
        task_dataframes[task] = pd.DataFrame({
            "elmo_predictions": all_predictions[elmo_key],
            "transformer_predictions": all_predictions[transformer_key]
        })

# Load ground truths
ground_truths = load_ground_truths()

# Add ground truths to the DataFrames
for task, df in task_dataframes.items():
    if task in ground_truths:
        df["ground_truth"] = ground_truths[task]
    else:
        print(f"No ground truth available for task: {task}")

# Example: Display updated DataFrame for a specific task
for task, df in task_dataframes.items():
    print(f"Task: {task}")
    print(df.head())


# Initialize a list to store Cohen's d values for each task
cohen_d_values = []

# Calculate absolute differences, mean absolute errors, and Cohen's d
for task, df in task_dataframes.items():
    if "ground_truth" in df.columns:
        # Absolute error columns
        df["elmo_absolute_error"] = abs(df["elmo_predictions"] - df["ground_truth"])
        df["transformer_absolute_error"] = abs(df["transformer_predictions"] - df["ground_truth"])
        
        # Mean absolute error columns
        df["elmo_mae"] = df["elmo_absolute_error"].mean()
        df["transformer_mae"] = df["transformer_absolute_error"].mean()
        
        # Cohen's d for absolute error difference
        task_cohen_d = cohen_d(df["elmo_absolute_error"], df["transformer_absolute_error"])
        df["cohen_d"] = task_cohen_d
        
        # Append Cohen's d to the list
        cohen_d_values.append(task_cohen_d)
    else:
        print(f"Ground truth not found for task: {task}")


# Example: Display updated DataFrame for a specific task
for task, df in task_dataframes.items():
    print(f"Task: {task}")
    print(df.head())
    

# Calculate the average Cohen's d across all tasks
average_cohen_d = np.nanmean(cohen_d_values)  # Use np.nanmean to handle NaN values
print(f"Average Cohen's d across all datasets: {average_cohen_d}")