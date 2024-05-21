import re
import time
import warnings

import os
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd

from pandas import DataFrame
from scipy import stats
from scipy.stats import pearsonr

import en_core_web_md
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning

from dissertation.preprocessing import prepare_ml_data

from dissertation.config import *


nlp = en_core_web_md.load()

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def get_model(model_name):
    
    assert model_name in ml_models, 'model_name should be in config'
    
    if 'elasticnet' in model_name.lower():
        return ElasticNet()
    if 'randomforest' in model_name.lower():
        return RandomForestRegressor()
    

def regression_grid_train_test_report(model, x_train, y_train, x_val, x_test, y_test,
                                      paramater_grid, cv, method, score = 'explained_variance', model_output_dir = None, n_iter = 60):

    global frame

    #start timer
    start = time.time()

    print(
        '\n Performing grid search....hold tight... \n ============================='
    )

    model_name = model
    construct = y_test.name

    if model_output_dir:
        path = f'{model_output_dir}/{method}/{construct}.pkl'

    ###### grid search

    # construct grid search
    # number of parameter settings set to 60
    
    gs = RandomizedSearchCV(model,
                            param_distributions=paramater_grid,
                            scoring=score,
                            cv=5,
                            verbose=3,
                            n_iter=n_iter,
                            random_state=152,
                            n_jobs=-1)

    #fit on training data
    gs.fit(x_train, y_train)
    best_parameters = gs.best_params_
    best_estimator = gs.best_estimator_

    print('Grid Search Complete')
    print('==================================')

    ##### predict on test and validation data
    y_pred_test = best_estimator.predict(x_test)
    y_pred_val = best_estimator.predict(x_val)

    ##### savem best model
    if model_output_dir:
        joblib.dump(best_estimator, path)
        print('Best model saved')

    ###### regression report

    print(f'Outcome Variable: {construct}')

    #number of grid search combinations

    n_iterations = 1

    for value in paramater_grid.values():
        n_iterations *= len(value)

    print(f'Number of original grid search combinations: {n_iterations}')

    print(f'Best parameters for {model_name} were {best_parameters}')

    print('\n Results Below')

    # performance on test set for reporting
    r = pearsonr(y_test, y_pred_test)[0]
    print('r: ', r)
    print('==================================')

    #results data frame

    frame = pd.DataFrame([[construct, method, model_name, r]],
                         columns=['construct', 'method', 'model_name', 'r'])

    end = time.time()

    time_elapsed = (end - start) / 60

    print(f'Time Elapsed: {time_elapsed} minutes')

    print('\n \n \n Analysis Complete')

    return {'frame': frame, 'r': r, 'y_test_pred': y_pred_test, 'y_val_pred': y_pred_val, 'construct': construct}


def train_test_loop_baseline(models, x_train, y_train_list, x_val,
                             x_test, y_test_list, method, output_dir, cv = 5):

    dfs = []
    predictions = {}
    construct_list = []
    
    save_dir = f'{output_dir}/{method}'

    for model_name in models:
        print(f'running {model_name}')
        model = get_model(model_name)
        paramater_grid = ml_param_grid[model_name]
        
        for i in tqdm(range(0, len(y_train_list))):
            run_results = regression_grid_train_test_report(
                model, x_train, y_train_list[i], x_val, x_test, y_test_list[i],
                paramater_grid, cv, method)
            # predictions are on the validation data for ensembling
            # performance is on the test data
            frame, r, y_pred, construct = run_results['frame'], run_results['r'], run_results['y_val_pred'] , run_results['construct']
            dfs.append(frame)
            # save predictions
            predictions[construct] = y_pred
            construct_list.append(construct)
        
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(f'{save_dir}_predictions.csv')

    output = pd.concat(dfs)

    output['construct'] = construct_list
    print(f'Results...\n {output}')
    output.to_csv(f'{save_dir}_results.csv')
    print('baselines finished')
    return output

def ml_predict(path, x_test):

    # Load the saved model for predictions
    loaded_model = joblib.load(path)

    ##### predict on test data using the loaded model
    y_pred = loaded_model.predict(x_test)

    return y_pred

def get_all_model_paths(path):
    return os.lisdir(path)

def all_ml_predictions(root_path,
                       x_test,
                       method,
                       root=''):

    paths = os.listdir(root_path)

    predictions = {}

    #eacon
    for path in paths:
        preds = ml_predict(f'{root}{method}/{path}', x_test)
        construct = path.split('_')[0]
        predictions[construct] = preds

    return predictions

def run_ml(path, output_dir):
    
        ml_data = prepare_ml_data(path)
        

        bow_results = train_test_loop_baseline(ml_models, ml_data['bow_x_train'], ml_data['y_train_list'], ml_data['bow_x_valid'], ml_data['bow_x_test'], ml_data['y_test_list'],
                                 'bow', output_dir)
        
        empath_results = train_test_loop_baseline(ml_models, ml_data['empath_x_train'], ml_data['y_train_list'],  ml_data['empath_x_valid'], ml_data['empath_x_test'], ml_data['y_test_list'],
                                 'empath', output_dir)
        
        print(f'{bow_results} \n \n{empath_results}')