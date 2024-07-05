import pandas as pd

df = pd.read_csv('/Users/I745133/Desktop/git/NLP/dissertation/output/example_steiger_format.csv')

print(df.corr())

df.corr().to_csv('/Users/I745133/Desktop/git/NLP/dissertation/output/results/example_corr.csv')