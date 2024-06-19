# blip2를 이용해 captioning을 한 데이터를 바탕으로 qa 데이터를 생성하는 코드입니다.
import csv
import re
import json
import base64
import mimetypes
from openai import OpenAI

client = OpenAI(
    api_key=open('key.txt', 'r').read(),
)

#생성된 QA를 csv파일로 작성하는 함수
def get_QA(id,caption, Q, A):
    f = open('gptData.csv','a',newline='')
    wr = csv.writer(f)
    wr.writerow([id,caption,Q,A])
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

#프롬프트를 전달하면 GPT 4o의 응답을 받는 함수
def get_gpt_response_for4(prompt, caption, image):
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
                    "content": [
                        {"type": "text", "text": "class of drawing " + caption},
                        {"type": "image_url","image_url": {"url": image}}
                    ]
                }
            ],
            model="gpt-4o",
        )
    
    dst = response.choices[0].message.content

    return dst

def get_gpt_response_for4_nocaption(prompt, image):
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
                    "content": [
                        {"type": "image_url","image_url": {"url": image}}
                    ]
                }
            ],
            model="gpt-4o",
        )
    
    dst = response.choices[0].message.content

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

#gpt 4를 위한 프롬프트를 작성하는 함수
def get_prompt_for4():

    instruction = """You are a chatbot who makes QA with caption data. If you're given captions, please create 3 creative QAs for them. The captions are for a drawing with black pen on a white background. The template must be in json form. For example, 
    {"Q" : "[null,null,null]", "A" : "[null,null,null]"}
    You just have to fill in the null here.

    When generating questions and answers, refer to the delivered image to generate questions and answers.
    """
    example = ""

    prompt = instruction + example

    return prompt

#gpt 4를 위한 자세한 디렉션이 있는 프롬프트를 작성하는 함수
def get_prompt_for4_direction():

    instruction = """You are a chatbot who makes QA with caption data. If you're given captions, please create 3 creative QAs for them. The captions are for a drawing with black pen on a white background. The template must be in json form. For example, 
    {"Q" : "[null,null,null]", "A" : "[null,null,null]"}
    You just have to fill in the null here.

    Be aware that it must be exactly the same as the format given
    given format : {"Q" : "["Q1","Q2","Q3"]", "A" : "["A1","A2","A3"]"}

    When generating questions and answers, refer to the delivered image to generate questions and answers.

    And when forming QA, you should not ask questions related to the color of the drawing.

    In the drawing, it is important to ask what accessories the object is wearing or not. It does not have to be an accessory, and questions about the object and the object expressed in the drawing are preferred. Whether there is an object or not, what object it is in detail (e.g., cap, fedora, sun cap, ball cap, etc.), where it is, etc. Just refer to the explanation, and write a question creatively. You don't have to focus too much on accessories.
    """
    example = ""

    prompt = instruction + example

    return prompt

#gpt 4를 위한 자세한 디렉션이 있는 프롬프트를 작성하는 함수 - 5개
def get_prompt_for4_direction_5():

    instruction = """You are a chatbot who makes QA with caption data. If you're given captions, please create 5 creative QAs for them. The captions are for a drawing with black pen on a white background. The template must be in json form. For example, 
    { "Q" : "[null,null,null,null,null]", "A" : "[null,null,null,null,null]" }
    You just have to fill in the null here.

    Be aware that it must be exactly the same as the format given
    given format : { "Q" : "["Q1","Q2","Q3","Q4","Q5"]", "A" : "["A1","A2","A3","A4","A5"]" }

    When generating questions and answers, refer to the delivered image to generate questions and answers.

    And when forming QA, you should not ask questions related to the color of the drawing.

    In the drawing, it is important to ask what accessories the object is wearing or not. It does not have to be an accessory, and questions about the object and the object expressed in the drawing are preferred. Whether there is an object or not, what object it is in detail (e.g., cap, fedora, sun cap, ball cap, etc.), where it is, etc. Just refer to the explanation, and write a question creatively. You don't have to focus too much on accessories.
    """
    example = ""

    prompt = instruction + example

    return prompt

#gpt 4를 위한, 캡션이 없는 프롬프트를 작성하는 함수
def get_prompt_for4_nocaption():

    instruction = """You are a chatbot who makes QA. please create 3 creative QAs. The template must be in json form. For example, 
    {"Q" : "[null,null,null]", "A" : "[null,null,null]"}
    You just have to fill in the null here.
    Be aware that it must be exactly the same as the format given
    given format : {"Q" : "["Q1","Q2","Q3"]", "A" : "["A1","A2","A3"]"}

    When generating questions and answers, refer to the delivered image to generate questions and answers.
    """
    example = ""

    prompt = instruction + example

    return prompt

#gpt 4를 위한, 자세한 디렉션이 있는 프롬프트를 작성하는 함수
def get_prompt_for4_nocaption_direction():

    instruction = """You are a chatbot who makes QA. please create 3 creative QAs. The template must be in json form. For example, 
    {"Q" : "[null,null,null]", "A" : "[null,null,null]"}
    You just have to fill in the null here.
    Be aware that it must be exactly the same as the format given
    given format : {"Q" : "["Q1","Q2","Q3"]", "A" : "["A1","A2","A3"]"}

    When generating questions and answers, refer to the delivered image to generate questions and answers.

    And when forming QA, you should not ask questions related to the color of the drawing.

    In the drawing, it is important to ask what accessories the object is wearing or not. It does not have to be an accessory, and questions about the object and the object expressed in the drawing are preferred. Whether there is an object or not, what object it is in detail (e.g., cap, fedora, sun cap, ball cap, etc.), where it is, etc. Just refer to the explanation, and write a question creatively. You don't have to focus too much on accessories.
    """
    example = ""

    prompt = instruction + example

    return prompt

#예외 데이터를 기록하는 함수
def get_exception(id,caption,data):
    f = open('exception.csv','a',newline='')
    wr = csv.writer(f)
    wr.writerow([id,caption, data])
    f.close()

def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

'''
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
'''

for i in read_data('data/blip2_captioning.csv', 1)[1:]:
    print(i[0])
    base64_image = encode_image("C:/Users/User/Desktop/QA/data/sketch/original/train/"+i[0]+".png")
    drawingclass = i[0].split('(')[0]
    print(drawingclass)
    response = get_gpt_response_for4(get_prompt_for4_direction_5(), drawingclass, base64_image)
    print(response)
    pattern = r'\{[^}]*\}'
    res = re.findall(pattern, response)
    print(res)
    if len(res)==1:
        try:
            json_data = json.loads(res[0])
            get_QA(i[0],i[1],json_data["Q"][0],json_data["A"][0])
            get_QA(i[0],i[1],json_data["Q"][1],json_data["A"][1])
            get_QA(i[0],i[1],json_data["Q"][2],json_data["A"][2])
            get_QA(i[0],i[1],json_data["Q"][3],json_data["A"][3])
            get_QA(i[0],i[1],json_data["Q"][4],json_data["A"][4])
        except:
            get_exception(i[0],i[1], response)
    elif len(res)==5:
        try:
            for j in res:
                json_data = json.loads(j)
                get_QA(i[0],i[1],json_data["Q"],json_data["A"])
        except:
            get_exception(i[0],i[1], response)
    else:
        get_exception(i[0],i[1], response)
