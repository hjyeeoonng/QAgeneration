import pandas as pd
df = pd.read_csv('gptData0627_sorted.csv', header=None)
data = df.values.tolist()
cnt = 0
before = ""
for i in data:
    if i[0]!=before:
        cnt=0
    else:
        cnt+=1
    if cnt>=5:
        print(i)
        break
    before = i[0]