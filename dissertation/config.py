from ray import tune
import torch.nn as nn

transformer_config = {
    'model': 'google/bigbird-roberta-base',
    'tokenizer': 'google/bigbird-roberta-base'
}

ml_models = ['ElasticNet', 'RandomForestRegressor']

ml_param_grid = {
    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'max_iter': [1000, 5000, 10000]
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
        'max_depth': [10, 20, 30, 40, 50, 60],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'XGBRegressor': {
        'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'reg_alpha': [0, 0.1, 0.5, 1, 2, 5, 10],
        'reg_lambda': [0, 0.1, 0.5, 1, 2, 5, 10]
    }
}

nn_search_space =  {
    "hidden_size": tune.choice([10, 20, 30, 40, 50]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "dropout_rate": tune.uniform(0.0, 0.5),
    "activation_function": tune.choice([nn.ReLU, nn.Tanh])
}