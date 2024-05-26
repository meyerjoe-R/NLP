import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from scipy.stats import pearsonr
from dissertation.config import nn_search_space, ml_param_grid, ml_models

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

from dissertation.ML import get_model, regression_grid_train_test_report, correlation_scorer

from dissertation.preprocessing import prepare_train_test_data

######### Pre processing #########

scales = [
        'E_Scale_score', 'A_Scale_score', 'O_Scale_score', 'C_Scale_score',
        'N_Scale_score'
    ]

def drop_text(df):
    drop = ["open_ended_1", "open_ended_2", "open_ended_3", "open_ended_4", "open_ended_5"]
    return df[[col for col in df.columns if col not in drop]]

def gather_targets(scales, path='dissertation/data/MLdata.csv'):
    dataset_types = ['Dev', 'Test']
    
    df = pd.read_csv(path)
    valid_data = df.loc[df.Dataset == 'Dev']
    test_data = df.loc[df.Dataset == 'Test']
    train_data = df.loc[df.Dataset == 'Train']
    
    train_data = drop_text(train_data)
    test_data = drop_text(test_data)
    valid_data = drop_text(valid_data)

    # for QA
    train_data.to_csv('dissertation/data/train.csv')
    valid_data.to_csv('dissertation/data/valid.csv')
    test_data.to_csv('dissertation/data/test.csv')

    valid_targets = {scale: valid_data[scale].reset_index(drop=True) for scale in scales}
    test_targets = {scale: test_data[scale].reset_index(drop=True) for scale in scales}
    
    return valid_targets, test_targets

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
    
    pred_keyword = f'{prediction_type}_predictions'
    print(f'gathering {pred_keyword}')
    
    files = os.listdir(directory)
    pred_files = [file for file in files if pred_keyword in file]
    
    pred_dfs = [pd.read_csv(os.path.join(directory, file)) for file in pred_files]
    concatenated_dfs = {}

    for i, df in enumerate(pred_dfs):
        df = df.rename(columns=rename_dict)
        pred_dfs[i] = df

    for scale in scales:
        concatenated_dfs[scale] = pd.concat([df[scale] for df in pred_dfs], axis=1).reset_index(drop=True)

    print(f'concatenated data dict: {concatenated_dfs.keys()}')
    
    if targets_path:
        valid_targets, test_targets = gather_targets(scales)
        
        for scale in scales:
            if prediction_type == 'valid':
                concatenated_dfs[scale]['target'] = valid_targets[scale]
            else:
                concatenated_dfs[scale]['target'] = test_targets[scale]
                    
    os.makedirs(ensemble_directory, exist_ok=True)

    for scale, df in concatenated_dfs.items():
        df.to_csv(f'{ensemble_directory}/{scale}_{prediction_type}.csv', index=False)

    return concatenated_dfs

######## NN Ensemble ########

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
def objective(config, X, y, input_size=9, output_size=1, num_epochs=50):
    model = Ensemblenn(input_size, config['hidden_size'], output_size, activation_function=config["activation_function"], dropout_rate=config["dropout_rate"])
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # Ensure they are torch tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X.values, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y.values, dtype=torch.float32)

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
            all_targets.extend(targets.numpy().flatten())
            all_outputs.extend(outputs.numpy().flatten())

    correlation, _ = pearsonr(all_outputs, all_targets)
    return {'correlation': correlation}

def run_ray(tune, search_space, X, y, input_size=9):
    analysis = tune.run(
        tune.with_parameters(objective, X=X, y=y, input_size=input_size),
        config=search_space,
        num_samples=60,
        metric="correlation",
        mode="max"
    )
    print(analysis)
    # best hypers
    best_config = analysis.best_config
    return best_config

def train_model(input_size, hidden_size, output_size, best_config, X, y, num_epochs = 100):
    """
    to be used to retrain model on all data
    """
    
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X.values, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y.values, dtype=torch.float32)
        
    y = y.view(-1, 1)
    
    model = Ensemblenn(input_size, hidden_size, output_size,
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

def make_predictions(model, X_test):
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    return predictions

def calculate_correlation(predictions, y_test):
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Ensure predictions and y_test are 1D arrays
    predictions = predictions.flatten()
    y_test = y_test.flatten()
    
    correlation, _ = pearsonr(predictions, y_test.numpy())
    return correlation

def nn_ensemble(search_space, X_train, y_train, X_test, y_test, input_size=10, num_epochs=50, output_size = 1):
    best_config = run_ray(tune, search_space, X_train, y_train, input_size=input_size)
    model = train_model(input_size=input_size, output_size=output_size, best_config=best_config, X=X_train, y=y_train, num_epochs=num_epochs)
    predictions = make_predictions(model, X_test)
    final_correlation = calculate_correlation(predictions, y_test)
    return best_config, model, predictions, final_correlation

def tune_nn_loop(ensemble_directory, search_space):
    
    valid_dfs, test_dfs = gather_ensemble_data(ensemble_directory)

    for key in valid_dfs:
        print(f'{key}\n \n')
        valid_df = valid_dfs[key]
        test_df = test_dfs[key]
        
        print(f'valid data: \n {valid_df.head()}')
        print(f'test data: \n {test_df.head()}')

        X_train, y_train = get_x_y(valid_df)
        X_test, y_test = get_x_y(test_df)
        
        input_size = len([col for col in X_train.columns if 'target' not in col])
        
        print(f'initializing with {input_size} input_sze')
        
        nn_ensemble(search_space, X_train, y_train, X_test, y_test, input_size=input_size)
        
def train_test_nn():
    
    model = train_model(10, 50, 1, best_config, X_train,y_train, num_epochs = 20)
    preds = make_predictions(model, X_test)
    corr = pearsonr(preds.flatten(), y_test.values.flatten())
    print(corr)

########## xgboost ##########

def tune_xgb(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

    grid_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=ml_param_grid['XGBRegressor'], 
                                    scoring=make_scorer(correlation_scorer, greater_is_better=True), cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    correlation, _ = pearsonr(predictions, y_test)
    print("Best hyperparameters found: ", grid_search.best_params_)
    print("Correlation on test set: ", correlation)

    return predictions, correlation

def get_x_y(df):
    
    x_col = [col for col in df if 'target' not in col]
    x = df[x_col]
    y = df['target']
    return x, y

def clean_file_name(file):
    
    return file.split('/')[-1].replace('.csv', '').replace('_valid', '').replace('_test', '')


def gather_ensemble_data(ensemble_directory):
    
    files = os.listdir(ensemble_directory)
    print(f'ensemble files: {files}')
    
    valid_dfs = {clean_file_name(file): pd.read_csv(os.path.join(ensemble_directory, file)) for file in files if 'valid' in file}
    test_dfs = {clean_file_name(file): pd.read_csv(os.path.join(ensemble_directory, file)) for file in files if 'test' in file}
    
    return valid_dfs, test_dfs
    
def tune_xgb_loop(ensemble_directory, output_dir):
    # Load all validation and test datasets

    results = []
    
    valid_dfs, test_dfs = gather_ensemble_data(ensemble_directory)

    for key in valid_dfs:
        print(f'{key}\n \n')
        valid_df = valid_dfs[key]
        test_df = test_dfs[key]
        
        print(f'valid data: \n {valid_df.head()}')
        print(f'test data: \n {test_df.head()}')

        X_train, y_train = get_x_y(valid_df)
        X_test, y_test = get_x_y(test_df)
        
        prediction, correlation = tune_xgb(X_train, y_train, X_test, y_test)

        # Save predictions to output directory
        output_path = os.path.join(output_dir, f"{key}_xgb_predictions.csv")
        pd.DataFrame(prediction, columns=['prediction']).to_csv(output_path, index=False)

        # Append correlation result
        results.append({'construct': key, 'r': correlation})

    # Save summary of correlations
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(output_dir, 'xgb_ensemble_results.csv'), index=False)
    
####### ML ########

def ml_ensemble(models,
                train_dict,
                test_dict,
                method,
                output_dir,
                cv=5):

    dfs = []
    test_predictions = {}
    construct_list = []

    print('beginning ensemble modeling............')

    for model_name in models:
        save_dir = f'{output_dir}/{method}'
        print(f'running {model_name}')
        model = get_model(model_name)
        parameter_grid = ml_param_grid[model_name]

        for key in tqdm(train_dict.keys()):
            train_df = train_dict[key]
            test_df = test_dict[key]
            
            X_train, y_train = get_x_y(train_df)
            x_test, y_test = get_x_y(test_df)

            run_results = regression_grid_train_test_report(model_name,
                model, X_train, y_train, x_test, y_test,
                parameter_grid, cv, method)

            # Extract results from run_results
            frame, r, construct, y_pred_test = (
                run_results['frame'],
                run_results['r'],
                run_results['construct'],
                run_results['y_test_pred']
            )
            
            # override construct
            construct = key
            dfs.append(frame)
            test_predictions[construct] = y_pred_test
            construct_list.append(construct)
                
        # test_pred_df = pd.DataFrame(test_predictions)
        # test_pred_df.to_csv(f'{save_dir}_test_predictions.csv')

    output = pd.concat(dfs)
    output['construct'] = construct_list
    print(f'Results...\n {output}')
    output.to_csv(f'{save_dir}_results.csv')
    return output

# tune_xgb_loop(ensemble_directory, output_dir)
# NN
# tune_nn_loop(ensemble_directory, nn_search_space)
# best_config = {'activation_function': nn.ReLU, 'batch_size': 32, 'dropout_rate': .1, 'lr': 1e-5}


####### Ensemble Loop #######
    
def ensemble_(directory = 'dissertation/output', ensemble_directory = 'dissertation/output/ensemble_data',
            output_dir = 'dissertation/output', ensemble_output_directory = 'dissertation/output/ensemble_results'):
    
    for data_type in ['valid', 'test']:
        prepare_ensemble(directory, data_type)
    

    
    valid_dfs, test_dfs = gather_ensemble_data(ensemble_directory)
    ml_ensemble(ml_models, valid_dfs, test_dfs, 'ensemble', ensemble_output_directory)
    
        
   