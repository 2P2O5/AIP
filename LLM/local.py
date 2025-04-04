# TODO 明天给我去下3B的qwen!
# shutdown -h now

from llama_cpp import Llama
import os

from typing import Generator
dirnow = os.path.dirname(__file__)
# 指定本地模型的路径
model_path = dirnow + "/qwen2.5-0.5b-instruct-q8_0.gguf"

# 加载模型
global llm, log, search_url
llm = None
log = None
search_url = None

def ready(_log, _SEARCH_URL):
    global llm, log, search_url
    llm = Llama(
        model_path=model_path,
        verbose=False,
        n_ctx=32768
    )
    log, search_url = _log, _SEARCH_URL

# user = My, system = system, assistant = AI

def split_text_backwards(text, group_size=6):
    # 从后向前分隔文本
    result = []
    while text:
        result.insert(0, text[-group_size:])  # 将最后的5个字符放到结果的最前面
        text = text[:-group_size]  # 删除已分割的字符
    return result

def run(prompt, others = [], stream = True): # yield
    global llm
    with open( dirnow + "/promot", encoding = "utf-8") as c:
        with open( dirnow + "/tools", encoding = "utf-8") as c2:
            return llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": c2.read()
                    },
                    {
                        "role": "user",
                        "content": "请你介绍一下你是谁？"
                    },
                    {
                        "role": "assistant",
                        "content": c.read()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }] + others,
                max_tokens=32768,
                stream=stream
            )

def unsplit(text, others = []):
    global llm, log, search_url
    runer = run(text, others, False)
    return runer['choices'][0]["message"]["content"]
def split(text, others = []):

    global llm, log, search_url
    ed = []
    last = ""
    toqq = False
    runer = run(text, others)
    for chunk in runer:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            print(delta['content'], end="|", flush=True)
            # 使用 yieid 返回 一个搞完自动下一个
            if delta['content'] in ["，","。","；","：","！","？","。。。","。。。。。。","，",'——','--']:
                if last + delta['content'] in ed:
                    last = ""
                    break
                ed.append(last + delta['content'])
                yield last + delta['content']
                last = ""
            else:
                last = last + delta["content"]

    if last != "":
        yield last


if __name__ == "__main__":
    ready(None, None)
    print("Start!")
    import time
    a = time.time()
    print(unsplit("帮我生成一个女孩图片。"))
    print(time.time() - a)

# sk-cf205b0177ba47e181107727cce04039