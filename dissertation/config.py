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
    }
}
