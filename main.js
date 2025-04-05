console.log("\x1B[1m" + String.raw`
    ---------------------------------------------------------------
     ________  ___  ________   
    |\   __  \|\  \|\   __  \  
    \ \  \|\  \ \  \ \  \|\  \ 
     \ \   __  \ \  \ \   ____\
      \ \  \ \  \ \  \ \  \___|
       \ \__\ \__\ \__\ \__\   
        \|__|\|__|\|__|\|__|    by 2p2o5 [https://github.com/2p2o5]
    ---------------------------------------------------------------
    `)
const net = require('net');
const WebSocket = require('ws');
const NodeMediaServer = require('node-media-server');
const { exec } = require('child_process');
const axios = require('axios');
const http = require('http');

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { readFileSync } = require('fs');
const yaml = require('js-yaml');
const fs = require('fs');

const mc_front = require("./front/mc/main.js").exports
// const mc_front = () => { };

// 读取 config.yaml 文件
let configYaml;
try {
    configYaml = yaml.load(fs.readFileSync('./config.yaml', 'utf8'));
} catch (e) {
    console.error(getTime("FAIL"), "Failed to load config.yaml:", e.message);
    process.exit(1);
}

// 提取端口配置
const ports = configYaml.front.ports;

// 提取 napcat 配置
const napcatHttpPort = ports['napcat-http'];
const napcatWebsocketPort = ports['napcat-websocket'];

// 提取其他配置信息
global.GROUP_ID = configYaml.backend.qq.group_id;
// console.log(configYaml)
// 配置信息
const config = {
    ports: {
        loggerTcp: ports.loggerTCP,
        websocket_show: ports['websocket-show'],
        tcp_backend: ports['tcp-backend'],
        rtmp: ports.rtmp,
        wsCtrl: ports['websocket-control'],
        ctrl_http: ports['http-control'],
        backend_qq: ports['backend-qq'],
    },
    rtmpConfig: {
        rtmp: {
            port: ports.rtmp,
            chunk_size: 10000,
            gop_cache: true,
            ping: 30,
            ping_timeout: 60
        }
    }
};

// 时间戳函数
function getTime(type = "INFO") {
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth() + 1; // 月份从0开始，需要加1
    const day = now.getDate();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const seconds = now.getSeconds();
    const milliseconds = now.getMilliseconds();

    let colorCode;
    switch (type) {
        case "WARN":
            colorCode = "\x1b[43m"; // 黄色背景
            break;
        case "FAIL":
            colorCode = "\x1b[41m"; // 红色背景
            break;
        case "INFO":
        default:
            colorCode = "\x1b[42m"; // 绿色背景
            break;
    }

    // 蓝色字体的ANSI转义码
    const blueFont = "\x1b[34m";
    const resetCode = "\x1b[0m\x1b[1m"; // 重置颜色

    // 格式化时间字符串，包含毫秒
    const timeString = `[${year}/${month}/${day} ${hours < 10 ? "0" : ""}${hours}:${minutes < 10 ? "0" : ""}${minutes}:${seconds < 10 ? "0" : ""}${seconds}.${[].concat("0".repeat(3 - (milliseconds + "").length))}${milliseconds}]`;

    // 输出带颜色的时间字符串
    return `${blueFont}${timeString}${resetCode} ${colorCode}${type}${resetCode}`;
}


// TCP日志服务器
function startLoggerServer(port) {
    const server = net.createServer((socket) => {
        socket.on("data", (data) => {
            console.log(getTime(data.toString().split("|")[0]), data.toString().split("|")[1]);
        });
    });

    server.listen(port, () => {
        console.log(getTime(), `[logger] Start logger server: ${port}`);
    });
}

function extractImageContent(text) {
    const result = {
        imageContent: [],
        remainingText: text
    };

    const regex = /<IMAGE>(.*?)<\/IMAGE>/gs; // 匹配 <IMAGE> ... </IMAGE>，支持多行内容
    let match;

    while ((match = regex.exec(text)) !== null) {
        result.imageContent.push(match[1].trim()); // 提取 <IMAGE> ... </IMAGE> 中的内容
    }

    // 去除所有 <IMAGE> ... </IMAGE> 的内容
    result.remainingText = text.replace(regex, '').trim();

    return result;
}

// 提取 WebSocket 连接逻辑为函数
function setupWebSocket(url, onMessage, onError) {
    const ws = new WebSocket(url);

    ws.on('open', () => {
        console.log(getTime("INFO"), `[mid] WebSocket client connected to ${url}`);
    });

    ws.on('message', onMessage);

    ws.on('error', (err) => {
        console.error(getTime("FAIL"), `[mid] WebSocket client error: ${err.message}`);
    });

    return ws;
}

// 优化 startControlServer 函数
function startControlServer(wsPort, wsCtrl, tcpPort, ctrlHttp) {
    // HTTP 控制服务器
    http.createServer((req, res) => {
        try {
            res.end(readFileSync(`./front/ctrl${req.url}`));
        } catch {
            res.end("404");
        }
    }).listen(ctrlHttp, () => {
        console.log(getTime(), `[http_ctrl] HTTP control server started on port ${ctrlHttp}`);
    });

    // WebSocket 服务器
    const wssShow = new WebSocket.Server({ port: wsPort, host: '0.0.0.0' }, () => {
        console.log(getTime(), `[ws_show] WebSocket server started on port ${wsPort}`);
    });
    const wssCtrl = new WebSocket.Server({ port: wsCtrl, host: '0.0.0.0' }, () => {
        console.log(getTime(), `[ws_ctrl] WebSocket server started on port ${wsCtrl}`);
    });

    let wsClient = null;
    let tcpClient = null;

    wssShow.on('connection', (ws) => {
        wsClient = ws;
        ws.on('message', (message) => {
            if (tcpClient?.writable) {
                tcpClient.write(message);
            }
        });
    });

    wssCtrl.on('connection', (ws) => {
        ws.on('message', (message) => {
            if (tcpClient?.writable) {
                tcpClient.write(message);
            }
        });
    });

    // TCP 中间服务器
    const tcpServer = net.createServer((socket) => {
        tcpClient = socket;
        socket.on("data", (data) => {
            if (wsClient?.readyState === WebSocket.OPEN) {
                wsClient.send(data);
            } else {
                console.log(getTime("WARN"), `[mid] [qq] -> [tcp] Backend closed.`);
            }
        });

        socket.on('error', (err) => {
            console.error(getTime("FAIL"), `[mid] TCP server error: ${err.message}`);
        });

        socket.on('close', () => {
            console.log(getTime("FAIL"), '[mid] TCP server connection closed');
        });
    });

    tcpServer.listen(tcpPort, () => {
        console.log(getTime(), `[mid] TCP server started on port ${tcpPort}`);
    });

    let running = false;
    global.userid = null; /* User id. 用来 @*/

    const qqTcpServer = net.createServer((socket) => {
        socket.on("data", (data) => {
            running = false;
            const text = data.toString("utf-8").split("_!_")[1];
            const out = extractImageContent(text);
            console.log(out);
            const _data = {
                "group_id": GROUP_ID,
                "message": [
                    {
                        "type": "at",
                        "data": {
                            "qq": global.userid, //all为艾特全体
                        }
                    },
                    {
                        "type": "text",
                        "data": {
                            "text": " " + out.remainingText,
                        }
                    },
                    out.imageContent.length > 0 ? {
                        "type": "image",
                        "data": {
                            "file": `https://image.pollinations.ai/prompt/${out.imageContent[0]}?width=1024&height=1024&model=flux&nologo=true`,
                        }
                    } : {
                        "type": "text",
                        "data": {
                            "text": "",
                        }
                    }
                ]
            };

            axios.post(`http://127.0.0.1:${napcatHttpPort}/send_group_msg`, _data)
                .then(response => {
                    console.log(getTime(), '响应数据:', typeof response.data, response.data);
                })
                .catch(error => {
                    console.error('发生错误:', error);
                });

        });

        socket.on('error', (err) => {
            console.error(getTime("FAIL"), `[mid] qq(TCP) server error: ${err.message}`);
        });

        socket.on('close', () => {
            console.log(getTime("FAIL"), '[mid] qq(TCP) server connection closed');
        });
    });

    qqTcpServer.listen(config.ports.backend_qq, () => {
        console.log(getTime(), `[mid] Start qq server (TCP) at ${tcpPort + 1}`);
    });

    // QQ WebSocket 客户端
    const qqWs = setupWebSocket(`ws://127.0.0.1:${napcatWebsocketPort}`, (data) => {
        data = JSON.parse(data);
        if (data["post_type"] === "message") {
            handleQQMessage(data, tcpClient);
        }
    });
}

// 提取 QQ 消息处理逻辑为函数
function handleQQMessage(data, tcpClient) {
    if (running) return console.log(getTime(), `[mid] [qq] This message is ignored.`);
    const { sender, raw_message } = data;
    const { user_id, nickname } = sender;
    global.userid = user_id;

    console.log(getTime(), `[mid] QQ ${user_id}:${nickname} says |${raw_message}|`);
    const messageParts = raw_message.split("[CQ:at,qq=1352624036]");
    if (messageParts.length > 1) {
        const message = messageParts.join("");
        running = true;
        if (tcpClient?.writable) {
            tcpClient.write(`//_run QQ的${nickname}说：${message}`);
        } else {
            running = false;
            console.log(getTime("WARN"), `[mid] [qq] -> [tcp] Backend closed.`);
        }
    }
}

function startRtmpServer(rtmpConfig) {
    const nms = new NodeMediaServer(rtmpConfig);
    nms.run();
    console.log(getTime(), `[rtmp] Start RTMP server on port ${rtmpConfig.rtmp.port}`);
}

function startFront() {
    let mainWindow = null;
    // mc_front(getTime);

    app.on('window-all-closed', () => {
        app.quit();
    });

    app.on('ready', () => {
        mainWindow = new BrowserWindow({
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: true,
                enableRemoteModule: true // 允许使用remote模块
            },
            frame: false,
            fullscreen: true
        });

        mainWindow.loadURL(`file://${path.join(__dirname, '/front/index.html')}#${config.ports.wsCtrl}`);
        mainWindow.focus();
    });
}


function startBackend() {
    exec("./.venv/bin/python main.py");
    console.log(getTime(), `[main] Send backend sign start`)
}

async function startFfmpeg() {
    exec("yarn ffmpeg");
};


// 启动所有服务
function startAllServices() {
    startLoggerServer(config.ports.loggerTcp);
    startControlServer(config.ports.websocket_show, config.ports.wsCtrl, config.ports.tcp_backend, config.ports.ctrl_http);
    startRtmpServer(config.rtmpConfig);
    startFront();
    setTimeout(() => {
        startBackend();
        startFfmpeg()
    }, 1000);

}

startAllServices();