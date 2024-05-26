import pandas as pd
import os
import matplotlib.pyplot as plt
import ast

from dissertation.confidence_interval import r_confidence_interval

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
    files = os.listdir(directory_path)
    print(f'analyzing results from: {files}')
    
    # Loop through all files in the directory
    for filename in files:
        # Check if 'results' is in the file name
        if 'results' in filename and filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            df.rename(columns=column_mapping, inplace=True)
            
            if 'model_name' not in df.columns:
                model_name = filename.replace('.csv', '').replace('_results', '')
                df['model_name'] = model_name
                
                # If the file is in wide format
                if set(column_mapping.values()).issubset(df.columns):
                    df = convert_wide_to_long(df, model_name)
                if 'transformer' in filename:
                    df['method'] = 'transformer'
                elif 'lstm' in filename:
                    df['method'] = 'lstm'
            # Append the DataFrame to the list
            dataframes.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    print(concatenated_df)
    concatenated_df.to_csv(os.path.join(output_dir, 'concatenated_results.csv'), index=False)
    return concatenated_df


######### Plotting #########
def plot_average_performance(file_path, output_dir, group_by_column, y_lim=(0, 1)):
    """
    Plots the average performance by a specified column and saves the plot as a .png file.
    
    :param file_path: Path to the input CSV file.
    :param output_dir: Directory where the plot will be saved.
    :param group_by_column: Column name to group by for calculating average performance.
    :param y_lim: Tuple specifying the y-axis limits. Default is (0, 1).
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Remove any unnamed columns that are unnecessary
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    if group_by_column != 'method':
        df = df[[col for col in df.columns if 'method' not in col]]
    
    # Extract and calculate the bounds of the 95% CI
    df[['ci_low', 'ci_high']] = df['95% CI'].apply(ast.literal_eval).apply(pd.Series)
    
    # Group by the specified column and calculate the average 'r' and the bounds of the 95% CI
    grouped = df.groupby(group_by_column).agg({
        'r': 'mean',
        'ci_low': 'mean',
        'ci_high': 'mean'
    }).reset_index()
    
    # Calculate the error bars (distance from the mean 'r' to the average bounds)
    grouped['ci_error_low'] = grouped['r'] - grouped['ci_low']
    grouped['ci_error_high'] = grouped['ci_high'] - grouped['r']
    
    # Determine plot settings based on the number of categories
    num_categories = len(grouped[group_by_column])
    if num_categories > 10:  # For many categories, use larger figure and rotate labels
        fig_size = (12, 8)
        rotation = 45
        wrap_labels = True
    else:  # For fewer categories, smaller figure and no rotation
        fig_size = (8, 6)
        rotation = 0
        wrap_labels = False
    
    # Adjust figure size for better readability
    plt.figure(figsize=fig_size)
    plt.bar(
        grouped[group_by_column], grouped['r'],
        yerr=[grouped['ci_error_low'], grouped['ci_error_high']],
        capsize=5, color='black', edgecolor='gray', error_kw=dict(ecolor='gray', lw=2)
    )
    plt.xlabel(group_by_column, fontsize=12, fontname='DejaVu Sans')
    plt.ylabel('Average Performance (r)', fontsize=12, fontname='DejaVu Sans')
    plt.title(f'Average Performance by {group_by_column}', fontsize=14, fontname='DejaVu Sans')
    
    # Wrap long labels and adjust rotation if necessary
    if wrap_labels:
        labels = [ '\n'.join(label.split('_')) for label in grouped[group_by_column]]
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=rotation, ha='right', fontsize=10, fontname='DejaVu Sans')
    else:
        plt.xticks(rotation=rotation, fontsize=10, fontname='DejaVu Sans')
    
    plt.yticks(fontsize=10, fontname='DejaVu Sans')
    plt.ylim(y_lim)  # Set y-axis limits
    plt.grid(False)  # Disable grid lines for a cleaner look
    plt.tight_layout()
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot as a .png file
    output_file = os.path.join(output_dir, f'average_performance_by_{group_by_column}.png')
    plt.savefig(output_file, dpi=300)  # Higher DPI for better quality
    plt.close()

    print(f'Plot saved to {output_file}')
    

def save_grouped_dataframes(file_path, output_dir, group_by_column):
    """
    Groups the data by the specified column and saves each group as a separate CSV file.

    :param file_path: Path to the input CSV file.
    :param output_dir: Directory where the grouped CSV files will be saved.
    :param group_by_column: Column name to group by ('construct' or 'method').
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group by the specified column and save each group to a separate CSV file
    for group_value, group_df in df.groupby(group_by_column):
        # Construct a valid filename
        group_value_name = group_value.replace(' ', '_').replace('/', '_')
        output_file = os.path.join(output_dir, f'{group_value_name}.csv')
        
        # Save the group DataFrame to a CSV file
        group_df.to_csv(output_file, index=False)
        print(f'Saved {group_by_column}={group_value} data to {output_file}')
    
def main(output_dir='dissertation/output/results',
         directory_path = 'dissertation/output', 
         results_path = 'dissertation/output/results/concatenated_results.csv', create_concatenated_data = False, add_CI = False,
        plot_average_by_construct=False,
        save_separate_dataframes = True):
    
    # create concatenated results across all models and constructs
    if create_concatenated_data:
        concatenate_test_files(directory_path)
        get_highest_r_by_construct(results_path)
        
    if add_CI:
        # add in confidence intervals
        results = pd.read_csv(results_path)
        results['95% CI'] = results['r'].apply(r_confidence_interval)
        results.to_csv(results_path)
    if plot_average_by_construct:
        group_bys = ['construct', 'method', 'model_name']
        for slice in group_bys:
            plot_average_performance(results_path, output_dir, slice)
    if save_separate_dataframes:
        group_bys = ['construct', 'method']
        for slice in group_bys:
            save_grouped_dataframes(results_path, output_dir, slice)
main()