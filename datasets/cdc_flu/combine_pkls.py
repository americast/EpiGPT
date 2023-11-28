import os
import pickle

files = [x for x in os.listdir() if x.endswith('.pkl')]
flu_all, covid_all = [], []
for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    feat_flu, feat_covid = data[:, 4], data[:, 3]
    flu_all.append([feat_flu[:-30], feat_flu[-30:]])
    covid_all.append([feat_covid[:-30], feat_covid[-30:]])

with open('cdc_flu.pkl', 'wb') as f:
    pickle.dump((flu_all, None), f)
with open('cdc_covid.pkl', 'wb') as f:
    pickle.dump((covid_all, None), f)