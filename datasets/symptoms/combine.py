import pandas as pd
import pickle
import pudb


full_data = []

df = pd.read_csv('df.csv')
symptoms = df.columns.tolist()[4:]
for state in df["geo_code"].unique().tolist():
    df_state = df[df["geo_code"] == state]
    for symptom in symptoms:
        k = df_state[symptom].tolist()
        full_data.append([k[:-30], k[-30:]]) # train, test

with open('symp.pkl', 'wb') as f:
    pickle.dump((full_data, None), f)


    