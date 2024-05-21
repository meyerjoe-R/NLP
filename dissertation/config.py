

transformer_config = {
    'model': 'google/bigbird-roberta-base',
    'tokenizer': 'google/bigbird-roberta-base'
}

ml_models = ['ElasticNet', 'RandomForestRegressor']

ml_param_grid = {'ElasticNet': {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1]},
                 'RandomForestRegressor': {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
    }
    }