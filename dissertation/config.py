from ray import tune
import torch.nn as nn

transformer_config = {
    'model': 'google/bigbird-roberta-base',
    'tokenizer': 'google/bigbird-roberta-base'
}

ml_models = ['XGBRegressor', 'CatBoostRegressor', 'LGBMRegressor', 'ElasticNet', 'RandomForestRegressor']

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
    },
    'CatBoostRegressor':  {
    'iterations': [100, 200, 300, 500, 1000],                # Number of boosting iterations
    'depth': [4, 6, 8, 10, 12],                             # Depth of the tree
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],          # Learning rate
    'l2_leaf_reg': [1, 3, 5, 7, 9],                         # L2 regularization term
    'border_count': [32, 64, 128],                          # The number of splits for numerical features
    'bagging_temperature': [0.5, 1, 2, 3],                  # Controls the intensity of Bayesian bagging
    'random_strength': [1, 2, 5, 10],                       # Strength of the random effect for trees
    'one_hot_max_size': [2, 5, 10],                         # Convert the feature to one-hot encoding if the number of different values is less than or equal to the given parameter value
    'rsm': [0.5, 0.8, 1],                                   # Random subspace method. Fraction of features to be used at each iteration
    'boosting_type': ['Ordered', 'Plain'],                  # Type of boosting
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'], # Tree growing policy
    'leaf_estimation_iterations': [1, 3, 5, 10],            # Number of gradient steps when calculating the values in leaves
    'verbose': [False]                                      # Output options
},
    
    'LGBMRegressor': {
    'n_estimators': [100, 200, 500, 1000],
    'num_leaves': [31, 50, 70, 100],
    'max_depth': [-1, 10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_child_samples': [20, 50, 100],
    'min_child_weight': [0.001, 0.01, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

}

nn_search_space =  {
    "hidden_size": tune.choice([10, 20, 30, 40, 50]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "dropout_rate": tune.uniform(0.0, 0.5),
    "activation_function": tune.choice([nn.ReLU, nn.Tanh])
}