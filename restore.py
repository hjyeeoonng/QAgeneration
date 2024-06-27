import pandas as pd
import csv
import os

df = pd.read_csv('sorted_gptData0625_2.csv', header=None)

# 특정 폴더 경로 설정
folder_path = 'C:/Users/User/Desktop/QA/data/sketch/original+acc/train'

# 폴더 내 파일명 가져오기
file_names = os.listdir(folder_path)

datas = df[0].tolist()

def writeData(id):
    f = open('lostData.csv','a',newline='')
    wr = csv.writer(f)
    wr.writerow([id])
    f.close()

for i in file_names:
    if i[:-4] in datas:
        pass
    else:
        writeData(i[:-4])