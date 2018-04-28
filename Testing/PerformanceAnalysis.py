import pandas as pd
import plotly
import os

gt = pd.read_csv('../Groundtruth/Groundtruth.csv')
cols = [col for col in gt.columns if 'UP' in col or 'DOWN' in col]
results = gt.groupby('Difficulty')[cols].sum()

for filename in os.listdir('../Results'):
    data = pd.read_csv('../Results/' + filename)
    df = pd.merge(gt, data, on='File', suffixes=('_GT', '_' + filename))
    cols = [col for col in df.columns if 'UP' in col or 'DOWN' in col]
    print('Average fps ' + filename, df['FPS'].mean())
    results['UP_' + filename] = df.groupby('Difficulty')[cols].sum()['UP_' + filename]
    results['DOWN_' + filename] = df.groupby('Difficulty')[cols].sum()['DOWN_' + filename]
    results['RESULT_' + filename] = 1-round(abs((results['UP_' + filename] + results['DOWN_' + filename]) - (results['UP'] + results['DOWN'])) /
                                          (results['UP'] + results['DOWN']), 2)

cols = [col for col in results.columns if 'RESULT' in col]
print(results)
print("============")
print(results[cols])

