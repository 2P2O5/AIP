

(async () => {
  // 等待页面加载并点击 canvas 元素
  // setTimeout(async () => {
  //   await page.click("canvas");
  // }, 1000);
  
  // 使用 ffmpeg 捕获屏幕并推流到 RTMP 服务器
  ffmpeg()
  .input(':10.0+0,0') // 捕获 Xvfb 虚拟显示器的屏幕
  .inputOptions([
    '-f x11grab', // 使用 x11grab 捕获屏幕
    '-framerate 24', // 设置帧率为 24
    '-video_size 1920x1080', // 设置视频分辨率，确保与 Xvfb 设置一致
    '-draw_mouse 0' // 不显示鼠标
  ])
  .outputOptions([
    '-f flv', // 输出为 flv 格式
    '-c:v libx264', // 使用 libx264 编码器（H.264）
    '-crf 0', // 无损压缩，保持原始画质
    '-preset ultrafast', // 使用 ultrafast 预设，减少 CPU 占用
    '-tune zerolatency', // 设置为零延迟模式，适用于流媒体推送
  ])
  .output('rtmp://127.0.0.1/Dhl') // 推流到 RTMP 服务器
  .on('start', (commandLine) => {
    console.log('Spawned Ffmpeg with command: ' + commandLine);
  })
  .on('end', () => {
    console.log('Finished processing');
  })
  .on('error', (err) => {
    console.error('Error:', err);
  })
  .run();
})();
