import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from scipy.stats import pearsonr

from dissertation.config import nn_search_space, ml_param_grid

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from dissertation.preprocessing import prepare_train_test_data

def gather_targets(path):
    # Load data and concatenate responses
    df = pd.read_csv(path)
    train, valid, test, y_train_list, y_val_list, y_test_list = prepare_train_test_data(df)
    
    return valid, y_val_list, test, y_test_list


def prepare_ensemble(directory, prediction_type, ensemble_directory='dissertation/output/ensemble_data', targets_path='dissertation/data/MLdata.csv'):
    
    assert prediction_type in ['valid', 'test']
    
    scales = [
        'E_Scale_score', 'A_Scale_score', 'O_Scale_score', 'C_Scale_score',
        'N_Scale_score'
    ]
    
    rename_dict = {
        'y_E_val': 'E_Scale_score', 'y_A_val': 'A_Scale_score', 
        'y_O_val': 'O_Scale_score', 'y_C_val': 'C_Scale_score', 'y_N_val': 'N_Scale_score'
    }
    
    # Determine the prediction type
    pred_keyword = f'{prediction_type}_predictions'
    
    files = os.listdir(directory)
    pred_files = [file for file in files if pred_keyword in file]
    pred_dfs = [pd.read_csv(os.path.join(directory, file)) for file in pred_files]

    concatenated_dfs = {}

    for i, df in enumerate(pred_dfs):
        # Rename columns based on rename_dict if they exist in the DataFrame
        # Due to minor error in saving transformer predictions
        df = df.rename(columns=rename_dict)
        pred_dfs[i] = df

    for scale in scales:
        concatenated_dfs[scale] = pd.concat([df[scale] for df in pred_dfs], axis=1).reset_index(drop=True)

    if targets_path:
        valid, y_val_list, test, y_test_list = gather_targets(targets_path)
        target_list = y_val_list if prediction_type == 'valid' else y_test_list
        for i, scale in enumerate(scales):
            if len(target_list) > i:
                concatenated_dfs[scale][f'target'] = target_list[i]
                    
    # Create the ensemble directory if it does not exist
    os.makedirs(ensemble_directory, exist_ok=True)

    # Saving the new DataFrames
    for scale, df in concatenated_dfs.items():
        df.to_csv(f'{ensemble_directory}/{scale}_{prediction_type}.csv', index=False)

    return concatenated_dfs


class Ensemblenn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function=nn.ReLU, dropout_rate=0.0):
        super(Ensemblenn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation_function()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Objective function
def objective(config, input_size=5, output_size=1, num_epochs=50):
    model = Ensemblenn(input_size, config['hidden_size'], output_size, activation_function=config["activation_function"], dropout_rate=config["dropout_rate"])
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation and calculation of correlation
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            all_targets.extend(targets.numpy())
            all_outputs.extend(outputs.numpy())

    correlation, _ = pearsonr(all_outputs, all_targets)
    tune.report(correlation=correlation)
    
def run_ray(tune, search_space):
    analysis = tune.run(
        objective,
        config=search_space,
        num_samples=60,
        metric="correlation",
        mode="max"
    )
    print(analysis)
    # best hypers
    best_config = analysis.best_config
    return best_config

def train_model(input_size, output_size, best_config, X, y, num_epochs = 50):
    """
    to be used to retrain model on all data
    """
    model = Ensemblenn(input_size, output_size,
                           activation_function=best_config["activation_function"],
                           dropout_rate=best_config["dropout_rate"])
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=best_config["lr"])

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=best_config["batch_size"], shuffle=True)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

##### xgboost #####

def correlation_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    correlation, _ = pearsonr(predictions, y)
    return correlation

def tune_xgb(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

    grid_search = RandomizedSearchCV(estimator=xgb_model, param_grid=ml_param_grid['XGBRegressor'], 
                                    scoring=make_scorer(correlation_scorer, greater_is_better=True), cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    correlation, _ = pearsonr(predictions, y_test)
    print("Best hyperparameters found: ", grid_search.best_params_)
    print("Correlation on test set: ", correlation)

    return predictions

def get_x_y(df):
    
    x_col = [col for col in df if 'target' not in col]
    x = df[x_col]
    y = df['target']
    return x, y

def tune_xgb_loop(ensemble_directory, output_dir):
    """
    Train on validation data, test on test data
    """
    # load all validation datasets
    files = os.listdir(ensemble_directory)
    valid_dfs = [pd.read_csv(os.path.join(ensemble_directory, file)) for file in files if 'valid' in file]
    test_dfs = [pd.read_csv(os.path.join(ensemble_directory, file)) for file in files if 'test' in file]
    
    # get data from original df
    train, valid, test, y_train_list, y_val_list, y_test_list = prepare_train_test_data(df)
    
    for valid_df, y_train, y_test in zip(valid_dfs, y_val_list,y_test_list, test_dfs):
        x, y = get_x_y(df)
        assert y_val_list[0] == y[0]
        prediction = tune_xgb(x, y_train, X_test, y_test)