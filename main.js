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

// WebSocket控制服务器
function startControlServer(wsPort, wsCtrl, tcpPort, crtlHttp) {
    http.createServer((res, rep) => {
        try {
            rep.end(readFileSync("./front/ctrl" + res.url));
        } catch {
            rep.end("404");
        }
    }).listen(crtlHttp, () => {
        console.log(getTime(), `[http_ctrl] Start websocket server on port ${crtlHttp}`);
    })
    const wss_show = new WebSocket.Server({ port: wsPort, host: '0.0.0.0' }, () => {
        console.log(getTime(), `[ws_show] Start websocket server on port ${wsPort}`);
    });
    const wss_ctrl = new WebSocket.Server({ port: wsCtrl, host: '0.0.0.0' }, () => {
        console.log(getTime(), `[ws_ctrl] Start websocket server on port ${wsCtrl}`);
    });
    let wsClient = null;
    let tcpClient = null;

    wss_show.on('connection', (ws) => {
        wsClient = ws;
        ws.on('message', (message) => {
            if (tcpClient && tcpClient.writable) {
                tcpClient.write(message);
            }
        });
    });

    wss_ctrl.on('connection', (ws) => {
        ws.on('message', (message) => {
            if (tcpClient && tcpClient.writable) {
                tcpClient.write(message);
            }
        });
    });

    const tcpServer = net.createServer((socket) => {
        tcpClient = socket;
        socket.on("data", (data) => {
            if (wsClient && wsClient.readyState === WebSocket.OPEN) {
                wsClient.send(data);
            } else {
                console.log(getTime("WARN"), `[mid] [qq] -> [tcp] Backend closed.`)
            }
        });

        socket.on('error', (err) => {
            console.error(getTime("FAIL"), `[mid] back(TCP) server error: ${err.message}`);
        });

        socket.on('close', () => {
            console.log(getTime("FAIL"), '[mid] back(TCP) server connection closed');
        });
    });

    tcpServer.listen(tcpPort, () => {
        console.log(getTime(), `[mid] Start mid server (TCP) at ${tcpPort}`);
    });


    let running = false;
    global.userid = null; /* User id. 用来 @*/

    const qq_tcpServer = net.createServer((socket) => {
        // TODO 文本分割 输出图片。
        // url = ()
        socket.on("data", (data) => {
            running = false;
            const text = data.toString("utf-8").split("_!_")[1]
            const out = extractImageContent(text);
            console.log(out)
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
                    }: {
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

    qq_tcpServer.listen(config.ports.backend_qq, () => {
        console.log(getTime(), `[mid] Start qq server (TCP) at ${tcpPort + 1}`);
    });

    const qq_ws = new WebSocket(`ws://127.0.0.1:${napcatWebsocketPort}`);

    // 连接成功后
    qq_ws.on('open', function open() {
        console.error(getTime("INFO"), `[mid] qq websocket client link natcat success!`);
    });
    // 接收到消息时
    qq_ws.on('message', function incoming(data) {
        data = JSON.parse(data);
        // "raw_message": "[CQ:at,qq=1352624036] 你好"
        if (data["post_type"] == "message") {
            if (running) return console.log(getTime(), `[mid] [qq] this message don't be use.`)
            var { sender, raw_message } = data;
            var { user_id, nickname } = sender;
            global.userid = user_id;
            console.log(getTime(), `[mid] qq ${user_id}:${nickname} say |${raw_message}|`)
            if ((a = raw_message.split("[CQ:at,qq=1352624036]")).length > 1) {
                let message = a.join("");
                running = true;
                if (tcpClient && tcpClient.writable) {
                    tcpClient.write(`//_run qq的${nickname}说：${message}`);
                } else {
                    running = false;
                    console.log(getTime("WARN"), `[mid] [qq] -> [tcp] Backend closed.`)
                }
            }
        }
    });

    // 处理错误
    qq_ws.on('error', function (err) {
        console.error(getTime("FAIL"), `[mid] qq websocket client error: ${err.message}`);
    });
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

        mainWindow.loadURL(`file://${path.join(__dirname, '/front/index.html')}`);
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
    // startRtmpServer(config.rtmpConfig);
    // startFront();
    setTimeout(() => {
        startBackend();
        // startFfmpeg()
    }, 1000);

}

startAllServices();