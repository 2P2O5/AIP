LLM_THREAD = 3
HOST = "127.0.0.1"
PLAYER = "可莉"
SEARCH_URL = "http://192.168.1.115:81/search?q="
GROUP_ID = "752683607"

import os
import sys
dirnow = os.path.dirname(__file__) + "/.venv/lib/python3.11/site-packages"
sys.path.append(dirnow)

import socket
try:
    logger = socket.socket()    
    logger.connect((HOST, 8000))
    logger.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log(data):
        if (type(data) != bytes):
            if (type(data) == str):
                data = bytes(data, encoding="utf-8")
            else:
                data = bytes(str(data),encoding="urf-8")

        logger.send(data)
        print(str(data, encoding="utf-8"))

    mid = socket.socket()
    mid.connect((HOST, 8002))
    mid.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    to_qq = socket.socket()
    to_qq.connect((HOST, 8003))
    to_qq.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
except:
    print("Mid.js don't work.")
    exit(1)

log("INFO|[server] Recv start sign")

try:
    import torch

except:
    log("FAIL|You are not in .venv")
    log("INFO|[server] Backend close.")
    exit(1)


try:
    import os
    import sys
    dirnow = os.path.dirname(__file__)
    sys.path.append(dirnow)
    import time
    # import TTS
    import LLM
    from concurrent.futures import ThreadPoolExecutor, wait

    threadPool = ThreadPoolExecutor(4)

    LLM.ready(log, SEARCH_URL)

    # 主循环
    log("INFO|[server] Back start!")
    while True:
        exits = True
        while exits:
            recv = str(mid.recv(1024), encoding="utf-8").split(" ")
            # 如果接收到结束指令
            if recv[0] == "":
                print("Mid.js don't work.")
                raise RuntimeError

            if recv[0] == "//run":
                text = " ".join(recv[1:])
                exits = False

            if recv[0] == "//_run":
                # 特许版本 不进行TTS.
                text = " ".join(recv[1:])
                log("INFO|[server] Start run (qq)"+text)
                out = "".join(LLM.unsplit(text))
                log("INFO|[server] retrun " + out)
                time.sleep(0.01)
                to_qq.send(bytes("_!_" + out, encoding="utf-8"))
                log("INFO|[server] Finish.")
                continue
    
        start_time = time.time()
        # 提交任务
        log("INFO|[server] Start run "+text)
        ou = list(LLM.split(text))

        log("INFO|[server] split " + str(len(ou)))
        ou_t = []
        for i in ou:
            ou_t.append(threadPool.submit(TTS.infer, i, "可莉"))
        mid.send(bytes("1", encoding="utf-8"))   
        time.sleep(0.01)  
        for i in range(len(ou)):
            wait([ou_t[i]])
            data = ou_t[i].result()
            mid.send(bytes("2", encoding="utf-8"))
            time.sleep(0.01)
            mid.send(bytes(f"_{ou[i]}", encoding="utf-8"))
            time.sleep(0.01)
            mid.send(TTS.np2wav([data]))
            time.sleep(0.01)
            mid.send(bytes("3", encoding="utf-8"))
            time.sleep(0.01)
            log("INFO|[server] Send tts part.")
        log("INFO|[server] Run time: "+str(time.time() - start_time) +"s.")
        mid.send(bytes("4", encoding="utf-8"))

except KeyboardInterrupt:
    log("INFO|[server] Backend close.")
    exit(0)
except BaseException as e:
    import traceback
    log("FAIL|[server] "+ str(e) + "\n" + traceback.format_exc())

try:
    import time
    time.sleep(0.01)
    log("INFO|[server] Backend close.")
except:
    pass