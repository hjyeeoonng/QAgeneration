import pandas as pd
data = "C:/Users/User/Desktop/QA/data/0627/gptData0627test.csv"

df = pd.read_csv(data, header=None)
df_filtered = df.iloc[:, 2:]
df_filtered.columns = ['input', 'response']
df_filtered.to_csv('filteringData.csv', index=False)
