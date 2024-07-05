import csv
from dissertation.config import ml_param_grid
import pandas as pd
import numpy as np

def hyperparameter_to_table(output_file = 'dissertation/output/tables/hyperparameter_grid.csv'):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Hyperparameter", "Values"])
        
        for model, params in ml_param_grid.items():
            for param, values in params.items():
                writer.writerow([model, param, values])
                
def concatenated_long_to_wide(file_path, output_file = 'dissertation/output/tables/wide_all_models.csv'):
    scales = [  
        'E_Scale_score', 'A_Scale_score', 'O_Scale_score', 'C_Scale_score',
        'N_Scale_score'
    ]    

    df = pd.read_csv(file_path)
    df['r'] = round(df['r'].astype(float), 2)

    # Combine r and 95% CI into one column for each construct
    df['r_with_CI'] = df.apply(lambda row: f"{row['r']} {row['95% CI']}", axis=1)

    # Pivot the DataFrame to make each construct a column with combined r and 95% CI
    wide_df = df.pivot(index=['method', 'model_name'], columns='construct', values='r_with_CI').reset_index()
    
    # round all numeric columns
    wide_df.to_csv(output_file)
    print(f'df saved to {output_file}')

concatenated_long_to_wide('/Users/I745133/Desktop/git/NLP/dissertation/output/results/concatenated_results.csv')