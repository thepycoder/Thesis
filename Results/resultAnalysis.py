import pandas as pd
import numpy as np

gnd = pd.read_csv('Annotations - GND.csv')
haar = pd.read_csv('Annotations - HAAR.csv')
hog = pd.read_csv('Annotations - HOG.csv')
net = pd.read_csv('Annotations - NET.csv')

temp = pd.merge(gnd, haar, on="File")
temp = pd.merge(temp, hog, on="File")
temp = pd.merge(temp, net, on="File")

grouped = temp.groupby(['Difficulty']).sum()

result = grouped.drop(grouped.columns[[4, 7, 10]], axis=1)

result['ACC_HAAR'] = (result['UP_HAAR'] + result['DOWN_HAAR']) / (result['UP'] + result['DOWN'])
result['ACC_HOG'] = (result['UP_HOG'] + result['DOWN_HOG']) / (result['UP'] + result['DOWN'])
result['ACC_NET'] = (result['UP_NET'] + result['DOWN_NET']) / (result['UP'] + result['DOWN'])

print(result)