# blip2를 이용해 captioning을 한 데이터를 바탕으로 qa 데이터를 생성하는 코드입니다.
import csv
import re
import json
from openai import OpenAI

client = OpenAI(
    api_key=open('key.txt', 'r').read(),
)

#생성된 QA를 csv파일로 작성하는 함수
def get_QA(caption, Q, A):
    f = open('gptData.csv','a',newline='')
    wr = csv.writer(f)
    wr.writerow([caption,Q,A])
    f.close()

#captioning 한 data를 읽어들이는 함수
def read_data(targetFileName, option=0):
    f = open(targetFileName,'r')
    data = csv.reader(f)

    if option == 0:
        #데이터를 라인별로 출력
        for line in data:
            print(line)
        return 0
    elif option == 1:
        #데이터를 라인별로 return 
        linedata = []
        for line in data:
            linedata.append(line)
        return linedata

#프롬프트를 전달하면 GPT 3.5의 응답을 받는 함수
def get_gpt_response(prompt, caption):
    response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "you have to make QA for a drawing."
                },
                {
                    "role": "assistant", 
                    "content": prompt
                },
                {
                    "role": "user", 
                    "content": caption
                }
            ],
            model="gpt-3.5-turbo",
        )
    
    dst = response.choices[0].message.content
    print(dst)

    return dst

#프롬프트를 작성하는 함수
def get_prompt():

    instruction = """You are a chatbot who makes QA with caption data. If you're given captions, please create 3 creative QAs for them. The captions are for a drawing with black pen on a white background. The template must be in json form. For example, 
    {"Q" : "[null,null,null]", "A" : "[null,null,null]"}
    You just have to fill in the null here.
    """
    example = ""

    prompt = instruction + example

    return prompt

#예외 데이터를 기록하는 함수
def get_exception(caption,data):
    f = open('exception.csv','a',newline='')
    wr = csv.writer(f)
    wr.writerow([caption, data])
    f.close()

for i in read_data('blip2_captioning.csv', 1)[1:]:
    print(i[1])
    response = get_gpt_response(get_prompt(), i[1])
    pattern = r'\{[^}]*\}'
    res = re.findall(pattern, response)
    print(res)
    if len(res)==1:
        try: 
            json_data = json.loads(res[0])
            get_QA(i[1],json_data["Q"][0],json_data["A"][0])
            get_QA(i[1],json_data["Q"][1],json_data["A"][1])
            get_QA(i[1],json_data["Q"][2],json_data["A"][2])
        except:
            get_exception(i[1], response)
    else:
        get_exception(i[1], response)