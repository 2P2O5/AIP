const mineflayer = require('mineflayer')
const mineflayerViewer = require('prismarine-viewer').mineflayer
const net = require("net");

/*
*
* A simple bot demo https://github.com/PrismarineJS/prismarine-viewer
* Start it then open your browser to http://localhost:3007 and enjoy the view
*
* changed by 2p2o5.
*/

exports.exports = (log) => {
    /* 创建tcp server 运行代码 */
    const bot = mineflayer.createBot({
        host: '127.0.0.1',
        port: '25565',
        username: 'Keli'
    })

    bot.once('spawn', () => {
        mineflayerViewer(bot, { port: 3007, firstPerson: true, viewDistance: 3 })
    })

    const server = net.createServer((socket) => {
        // socket.on("data", async (data) => {
        //     let code = eval(data.toString());
        //     console.log(log(), "Run code:");
        //     console.log(code, data);
        //     try{
        //         socket.write(await code(bot));
        //     }catch{

        //     }
        // });
    });

    server.listen(8004);
}

if (process.argv[2] != undefined) {
    exports.exports()
}