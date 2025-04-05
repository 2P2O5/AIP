import yaml
import os
import sys
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait

# 读取 config.yaml 文件
try:
    with open('./config.yaml', 'r', encoding='utf-8') as file:
        configYaml = yaml.safe_load(file)
except Exception as e:
    print(f"Failed to load config.yaml: {e}")
    exit(1)

# 提取配置
ports = configYaml['front']['ports']
GROUP_ID = configYaml['backend']['qq']['groupid']
LLM_THREAD = configYaml['backend']['LLM']['threads_num']
HOST = "127.0.0.1"
PLAYER = configYaml['backend']['MC']['player_name']
SEARCH_URL = "http://192.168.1.115:81/search?q="

# 添加 .venv 路径
dirnow = os.path.dirname(__file__)
venv_path = os.path.join(dirnow, ".venv/lib/python3.11/site-packages")
sys.path.append(venv_path)

# 初始化 socket 连接
def init_socket_connection(host, port, description):
    try:
        sock = socket.socket()
        sock.connect((host, port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock
    except Exception as e:
        print(f"Failed to connect {description}: {e}")
        exit(1)

# 初始化 logger
logger = init_socket_connection(HOST, ports['loggerTCP'], "loggerTCP")

def log(data):
    if not isinstance(data, bytes):
        data = bytes(str(data), encoding="utf-8")
    logger.send(data)
    print(data.decode("utf-8"))

# 初始化其他 socket
mid = init_socket_connection(HOST, ports['tcp-backend'], "tcp-backend")
to_qq = init_socket_connection(HOST, ports['backend-qq'], "backend-qq")

log("INFO|[server] Recv start sign")

def send_data(data):
    mid.send(bytes(data, encoding="utf-8"))
    time.sleep(0.01)
# 检查环境
try:
    import torch
except ImportError:
    log("FAIL|You are not in .venv")
    log("INFO|[server] Backend close.")
    exit(1)

# 加载模块
try:
    import TTS
    import LLM

    threadPool = ThreadPoolExecutor(4)

    tts = TTS.init(
        os.path.join(dirnow, "TTS/mods/chinese-roberta-wwm-ext-large"),
        os.path.join(dirnow, "TTS/mods/hoyoTTS/G_78000.pth"),
        os.path.join(dirnow, "TTS/mods/hoyoTTS/config.json"),
        device="cpu"
    )

    LLM.ready(log, SEARCH_URL)

    # 主循环
    log("INFO|[server] Back start!")
    while True:
        exits = True
        while exits:
            recv = mid.recv(1024).decode("utf-8").split(" ")
            if not recv[0]:
                raise RuntimeError("Mid.js don't work.")

            if recv[0] == "//run":
                text = " ".join(recv[1:])
                exits = False

            elif recv[0] == "//_run":
                text = " ".join(recv[1:])
                log(f"INFO|[server] Start run (qq) {text}")
                out = "".join(LLM.unsplit(text))
                log(f"INFO|[server] return {out}")
                time.sleep(0.01)
                to_qq.send(bytes("_!_" + out, encoding="utf-8"))
                log("INFO|[server] Finish.")
                continue

        start_time = time.time()
        log(f"INFO|[server] Start run {text}")
        ou = list(LLM.split(text))

        log(f"INFO|[server] split {len(ou)}")
        ou_t = [threadPool.submit(tts, i, "可莉") for i in ou]
        send_data("1")

        for i, future in enumerate(ou_t):
            wait([future])
            data = future.result()
            send_data("2")
            send_data(f"_{ou[i]}")
            send_data(data)
            send_data("3")
        
        log(f"INFO|[server] Run time: {time.time() - start_time}s.")
        send_data("4")

except KeyboardInterrupt:
    log("INFO|[server] Backend close.")
    exit(0)
except BaseException as e:
    log(f"FAIL|[server] {e}\n{traceback.format_exc()}")

try:
    time.sleep(0.01)
    log("INFO|[server] Backend close.")
except Exception:
    pass