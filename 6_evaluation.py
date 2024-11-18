# Using this command to run this .py file
# python ./6_evaluation.py ./cleaned_gold_label.csv -o ./evaluation.csv

import pandas as pd

df = pd.read_csv('cleaned_gold_label.csv')

# check 'published' and 'gold_label'
df['Given_acc'] = (df['published'] == df['cleaned_gold_label']).astype(int)
# compute accuracy between 'published' and 'gold_label'
df['given_acc'] = None 
df.loc[0, 'given_acc'] = df['Given_acc'].mean() * 100 

# check 'cleaned_prediction_date' and 'gold_label'
df['Our_prediction_acc'] = (df['cleaned_prediction_date'] == df['cleaned_gold_label']).astype(int)
# compute accuracy between 'cleaned_prediction_date' and 'gold_label'
df['our_prediction_acc'] = None 
df.loc[0, 'our_prediction_acc'] = df['Our_prediction_acc'].mean() * 100

df.to_csv('evaluation.csv', index=False)
