import pandas as pd
import plotly
import os

desired_width = 320
pd.set_option('display.width', desired_width)

foldername = "FPS"

gt = pd.read_csv('../Groundtruth/Groundtruth.csv')
cols = [col for col in gt.columns if 'UP' in col or 'DOWN' in col]
results = gt.groupby('Difficulty')[cols].sum()

for filename in os.listdir('../Analysis/' + foldername):
    data = pd.read_csv('../Analysis/' + foldername + "/" + filename)
    df = pd.merge(gt, data, on='File', suffixes=('_GT', '_' + filename))

    errors = []

    for index, row in df.iterrows():
        if row['UP_GT'] + row['DOWN_GT'] > 0:
            errors.append((abs(row['UP_' + filename] - row['UP_GT']) + abs(row['DOWN_' + filename] - row['DOWN_GT'])) / (row['UP_GT'] + row['DOWN_GT']))
        else:
            errors.append(abs(row['UP_' + filename] - row['UP_GT']) + abs(row['DOWN_' + filename] - row['DOWN_GT']))

    df['ERROR'] = errors

    print(df)
    print('=================================')
    print('Difficulty/Error of %s' % filename)
    print(df.groupby('Difficulty')['ERROR'].mean())
    print('=================================')