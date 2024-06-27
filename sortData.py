import pandas as pd
import re

df = pd.read_csv('gptData0627testwithobject_1.csv', header=None)
def extract_number(s):
    match = re.search(r'\((\d+)\)', s)
    return int(match.group(1)) if match else float('inf')

# 괄호 전의 텍스트를 추출하는 함수 정의
def extract_text(s):
    match = re.search(r'([a-zA-Z]+)\(\d+\)', s)
    return match.group(1) if match else s

# 첫 번째 컬럼을 텍스트와 숫자 기준으로 정렬
df_sorted = df.sort_values(by=[0], key=lambda x: x.apply(lambda s: (extract_text(s), extract_number(s))))

df_sorted.to_csv('gptData0627testwithobject_sorted.csv', index=False, header=False)