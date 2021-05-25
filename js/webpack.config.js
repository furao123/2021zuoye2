const path = require('path');
const os = require('os');
const HtmlWebpackPlugin = require('html-webpack-plugin');
//引入下载的插件

// 自动获取本地ip
function getNetworkIp() {
    // 打开的host
    let needHost = '';
    try {
        // 获得网络接口列表
        const network = os.networkInterfaces();
        for (const dev in network) {
            const iface = network[dev];
            for (let i = 0; i < iface.length; i++) {
                const alias = iface[i];
                if (alias.family === 'IPv4' && alias.address !== '127.0.0.1' && !alias.internal) {
                    needHost = alias.address;
                }
            }
        }
    }
    catch (e) {
        console.log(e);
        needHost = 'localhost';
    }
    return needHost;
}

module.exports = {
    mode: 'development',
    entry: {//定义脚本入口文件
        index: './index.js'
    },
    devtool: 'inline-source-map',
    devServer: { //设置服务器 host：本地ip port：端口号
        host: getNetworkIp(),
        port: 8866
    },
    plugins: [
        new HtmlWebpackPlugin({ //自动引入脚本文件，生成最终页面的插件
            filename: 'index.html',
            chunks: ['index'],
            template: 'template.html'
        })
    ],
    module: {
        rules: [
            {
                test: /\.js$/,
                loader: 'babel-loader', //把代码翻译成浏览器可识别的语言
                exclude: /node_modules/ 
            }
        ]
    },
    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist')
    }
};