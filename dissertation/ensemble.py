import os
import pandas as pd

def prepare_ensemble(directory):
    scales = [
        'E_Scale_score', 'A_Scale_score', 'O_Scale_score', 'C_Scale_score',
        'N_Scale_score'
    ]

    files = os.listdir(directory)
    pred_files = [file for file in files if 'prediction' in file]
    pred_dfs = [pd.read_csv(os.path.join(directory, file)) for file in pred_files]

    concatenated_dfs = {}
    
    for scale in scales:
        concatenated_dfs[scale] = pd.concat([df[scale] for df in pred_dfs], axis=0).reset_index(drop=True)

    # Saving the new DataFrames
    for scale, df in concatenated_dfs.items():
        df.to_csv(os.path.join(directory, f'ensemble_{scale}.csv'), index=False)

    return concatenated_dfs