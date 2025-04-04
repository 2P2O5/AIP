console.log(String.raw`
    ________  ___  ________   
   |\   __  \|\  \|\   __  \  
   \ \  \|\  \ \  \ \  \|\  \ 
    \ \   __  \ \  \ \  ____\
     \ \  \ \  \ \  \ \  \___|
      \ \__\ \__\ \__\ \__\   
       \|__|\|__|\|__|\|__|    by 2p2o5 [https://github.com/2p2o5]`);

const perfMonitor = {
    stats: {
        fps: 30,
        drawCalls: 0,
        triangles: 0,
        memory: 0
    },

    start(engine) {
        // 每分钟输出性能日志
        setInterval(() => {
            this.stats.fps = engine.getFps().toFixed(1);
            this.stats.memory = (performance.memory?.usedJSHeapSize / 1048576).toFixed(2) || 0;
            console.log(
                `[PERF] FPS: ${this.stats.fps} | ` +
                `DrawCalls: ${this.stats.drawCalls} | ` +
                `Tris: ${(this.stats.triangles / 1000).toFixed(1)}k | ` +
                `Memory: ${this.stats.memory}MB`
            );
        }, 3000); // 60秒间隔

        // 每帧收集数据
        engine.onEndFrameObservable.add(() => {
            this.stats.drawCalls = engine._drawCalls;
            this.stats.triangles = engine._renderingTriangleCount;
        });
    }
};
// ================= 全局配置 =================
const CONFIG = {
    CANVAS: true,
    SOUND: true,
    GRAVITY: new BABYLON.Vector3(0, -9.8, 0),
    FPS: 30,
    ANIMATION: {
        // 完整预加载清单（包含所有示例动画）
        PRELOAD_MANIFEST: [
            { type: "VmdLoader", path: "./mmd/Do/打招呼.pbv", use: "natural" },
            { type: "VmdLoader", path: "./mmd/Do/自然站立_00.pbv", use: "natural" },
            { type: "VmdLoader", path: "./mmd/Do/自然站立_01.pbv", use: "natural" },
            { type: "VmdLoader", path: "./mmd/Do/自然站立_02.pbv", use: "natural" },
            { type: "VmdLoader", path: "./mmd/Do/自然站立_03.pbv", use: "natural" },
            { type: "VmdLoader", path: "./mmd/Do/对话_00.pbv", use: "speak" },
            { type: "VmdLoader", path: "./mmd/Do/对话_01.pbv", use: "speak" },
            { type: "VmdLoader", path: "./mmd/Do/对话_02.pbv", use: "speak" },
            { type: "VmdLoader", path: "./mmd/Do/对话_03.pbv", use: "speak" },
            { type: "VmdLoader", path: "./mmd/Do/对话_04.pbv", use: "speak" },
            { type: "VmdLoader", path: "./mmd/Do/对话_05.pbv", use: "speak" }
        ]
    }
};

// ================= 核心应用类 =================
class MMDApplication {
    constructor() {
        this.mation_cache = new Map();
        this.audio_list = [];
        this.lastAnimationChange = 0;
        this._animationList = [
            CONFIG.ANIMATION.PRELOAD_MANIFEST.filter(a => a.use.includes("speak")),
            CONFIG.ANIMATION.PRELOAD_MANIFEST.filter(a => a.use.includes("natural"))
        ]
        this.init();
    }

    async init() {
        if (CONFIG.CANVAS) await this.initRenderer();
        this.initSocket();
        if (CONFIG.SOUND) this.initAudio();
        this.initEventHandlers();
    }

    // ================= 渲染器初始化 =================
    async initRenderer() {
        globalThis.HK = await HavokPhysics();
        this.canvas = document.getElementById("renderCanvas");
        this.engine = new BABYLON.Engine(this.canvas, true, {
            preserveDrawingBuffer: true,
            stencil: true,
            disableWebGL2Support: false,
            powerPreference: "high-performance",
            premultipliedAlpha: false
        });

        this.scene = await this.createScene();
        perfMonitor.start(this.engine);
        window.addEventListener("resize", () => this.engine.resize());
    }

    async createScene() {
        await BABYLON.Tools.LoadScriptAsync("./js/babylon.mmd.min.js");
        BABYLONMMD.SdefInjector.OverrideEngineCreateEffect(this.engine);

        const scene = new BABYLON.Scene(this.engine);
        scene.clearColor = new BABYLON.Color4(0, 0, 0, 0);

        // ================= 光照系统 =================
        scene.ambientColor = new BABYLON.Color3(1, 0.8, 0.8);
        const directionalLight = new BABYLON.DirectionalLight(
            "MainLight",
            new BABYLON.Vector3(0, -1, 1),
            scene
        );
        directionalLight.intensity = 1.0;
        directionalLight.position = new BABYLON.Vector3(0, 10, 10);

        // ================= 物理引擎 =================
        const havokPlugin = new BABYLON.HavokPlugin();
        scene.enablePhysics(CONFIG.GRAVITY, havokPlugin);

        // ================= MMD运行时 =================`
        this.mmdRuntime = new BABYLONMMD.MmdRuntime(
            scene,
            new BABYLONMMD.MmdPhysics(scene)
        );
        this.mmdRuntime.register(scene);

        // ================= 模型加载 =================
        this.engine.displayLoadingUI();

        const [modelMesh, _] = await Promise.all([
            BABYLON.loadAssetContainerAsync("./mmd/可莉2.0.pmx", scene).then((result) => {
                result.addAllToScene();
                return result.meshes[0];
            }),
            (async () => {
                const havokPlugin = new BABYLON.HavokPlugin();
                scene.enablePhysics(new BABYLON.Vector3(0, -98, 0), havokPlugin);
            })(),
        ]);

        this.mmdModel = this.mmdRuntime.createMmdModel(modelMesh);

        // ================= 动画预加载 =================
        await this.preloadAnimations(scene);
        this.engine.hideLoadingUI();

        // ================= 摄像头加载 =================
        scene.mmdCamera = new BABYLONMMD.MmdCamera("MmdCamera", new BABYLON.Vector3(0, 10, 0), scene);
        scene.mmdCamera.maxZ = 5000;
        return scene;
    }

    // ================= 动画预加载 =================
    async preloadAnimations(scene) {
        const loadTasks = CONFIG.ANIMATION.PRELOAD_MANIFEST.map(async ({ type, path }) => {
            const loader = await (new BABYLONMMD[type](scene).loadAsync("motion", path));
            this.mation_cache.set(path, loader);
        });
        await Promise.all(loadTasks);
    }

    // ================= 音频系统 =================
    async initAudio() {
        this.audio = new Audio();
        this.audio.addEventListener('ended', () => {
            URL.revokeObjectURL(this.audio.src);
            this.playing = false;
        });
    }

    // ================= 网络通信 =================
    async initSocket() {
        this.socket = new WebSocket("ws://127.0.0.1:8200");
        this.bufferTemp = [];
        this.text = "";

        this.socket.addEventListener("message", async (event) => {
            const data = event.data;
            if (data instanceof ArrayBuffer) {
                this.processBinaryData(data);
            } else {
                this.processTextData(await data.arrayBuffer());
            }
        });
    }

    processBinaryData(buffer) {
        switch (buffer.byteLength) {
            case 1:
                this.handleControlSignal(new Uint8Array(buffer)[0]);
                break;
            default:
                this.bufferTemp.push(buffer);
        }
    }

    handleControlSignal(code) {
        switch (String.fromCharCode(code)) {
            case '1': // 开始消息
                this.text = "";
                this.bufferTemp = [];
                break;
            case '2': // 开始文本
                this.recvText = true;
                break;
            case '3': // 合并音频
                this.mergeAudioBuffers();
                break;
            case '4': // 完成
                this.tag = 0;
                console.log("传输完成");
                break;
        }
    }

    // ================= 音频处理 =================
    mergeAudioBuffers() {
        const totalLength = this.bufferTemp.reduce((sum, buf) => sum + buf.byteLength, 0);
        const merged = new Uint8Array(totalLength);
        let offset = 0;

        this.bufferTemp.forEach(buf => {
            merged.set(new Uint8Array(buf), offset);
            offset += buf.byteLength;
        });

        this.audio_list.push(URL.createObjectURL(new Blob([merged.buffer])));
        this.bufferTemp = [];
    }

    // ================= 动画控制 =================
    async playNextAnimation() {
        if (!this.mmdRuntime || !this.scene) return;

        if (this.lastAnimationChange != this.mmdRuntime._currentFrameTime)
            return this.lastAnimationChange = this.mmdRuntime._currentFrameTime;

        const animationList = this._animationList[this.audio_list.length > 0 ? 1 : 0]
        const nextAnim = animationList[Math.floor(Math.random() * animationList.length)];

        this.mmdRuntime.pauseAnimation();
        this.mmdModel.removeAnimation(0);

        this.mmdModel.addAnimation(this.mation_cache.get(nextAnim.path));
        this.mmdModel.setAnimation("motion");
        this.mmdRuntime.seekAnimation(0);
        this.mmdRuntime.playAnimation();
    }

    // ================= 音频是否完成 =================
    checkAudioPlayback() {
        if (this.audio.paused && this.audio_list.length > 0) {
            const url = this.audio_list.shift();
            this.audio.src = url;
            this.audio.play();
            this.playing = true;
        }
    }

    // ================= 主循环 =================
    async initEventHandlers() {
        let lastRender = 0;
        const frameInterval = 1000 / CONFIG.FPS; // 每帧最小时间间隔(ms)
        
        this.mmdRuntime.timeScale = 60 / CONFIG.FPS;

        this.engine.runRenderLoop(() => {
            if (!this.scene || !this.scene.activeCamera) return;
            
            const now = performance.now();
            if (now - lastRender >= frameInterval) {
                this.scene.render();
                this.playNextAnimation();
                lastRender = now;
            }
            this.checkAudioPlayback();
        });
    }
}

// ================= 启动应用 =================
window.Main = new MMDApplication();