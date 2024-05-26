import pandas as pd
import os

column_mapping = {
    'y_A_val': "A_Scale_score",
    'y_C_val': "C_Scale_score",
    'y_E_val': "E_Scale_score",
    'y_N_val': "N_Scale_score",
    'y_O_val': "O_Scale_score"
}

def get_highest_r_by_construct(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group by 'construct' and find the row with the highest 'r' for each group
    highest_r_df = df.loc[df.groupby('construct')['r'].idxmax()].reset_index(drop=True)
    average_r = highest_r_df['r'].mean()
    print(f'average r: {average_r}')
    print(highest_r_df)
    return highest_r_df

def convert_wide_to_long(df, model_name):
    # Melt the wide format DataFrame to long format
    id_vars = ['model_name']
    value_vars = ['A_Scale_score', 'C_Scale_score', 'E_Scale_score', 'N_Scale_score', 'O_Scale_score']
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='construct', value_name='r')
    df_long['method'] = 'ensemble'  # Assuming method is 'ensemble', modify if needed
    return df_long

def concatenate_test_files(directory_path, output_dir='dissertation/output/results'):
    # Initialize an empty list to store individual DataFrames
    dataframes = []

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if 'results' is in the file name
        if 'results' in filename and filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            df.rename(columns=column_mapping, inplace=True)
            
            if 'model_name' not in df.columns:
                model_name = filename.replace('.csv', '')
                df['model_name'] = model_name
                
                # If the file is in wide format
                if set(column_mapping.values()).issubset(df.columns):
                    df = convert_wide_to_long(df, model_name)
                else:
                    df['method'] = 'ensemble'  # Assuming method is 'ensemble', modify if needed
            
            # Append the DataFrame to the list
            dataframes.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    print(concatenated_df)
    concatenated_df.to_csv(os.path.join(output_dir, 'concatenated_results.csv'), index=False)
    return concatenated_df
    
def main(file_path, output_dir='dissertation/output/results', directory_path = 'dissertation/output'):
    concatenate_test_files(directory_path)
    get_highest_r_by_construct(os.path.join(output_dir, 'concatenated_results.csv'))
    
main(file_path='dissertation/output/ensemble_results/ensemble_results.csv')