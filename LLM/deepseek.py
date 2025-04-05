
# 试试那个deepseek怎么样
import os
from openai import OpenAI

dirnow = os.path.dirname(__file__)

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

def ready(thread_num, _log, _search_url):
    pass

ROLE = open( dirnow + "/promot", encoding="utf-8").read()

def split(text, others = []):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": ROLE},
            {"role": "user", "content": text},
        ],
        stream=False
    ).choices[0].message.content
    last = ""
    for delta in response:
        print(delta, end="|", flush=True)
        if delta in ["，","。","；","：","！","？","。。。","。。。。。。","，",'——','--']:
            yield last + delta
            last = ""
        else:
            last = last + delta

    if last != "":
        yield last
