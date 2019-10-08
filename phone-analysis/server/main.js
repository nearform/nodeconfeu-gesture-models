process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const fs = require('fs');
const url = require('url');
const ws = require('ws');
const https = require('https');
const send = require('send');
const path = require('path');

const publicdir = path.resolve(__dirname, 'public');

// sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout server.key -out server.cer
const options = {
    key: fs.readFileSync(path.resolve(__dirname, 'server.key')),
    cert: fs.readFileSync(path.resolve(__dirname, 'server.cer'))
};

const server = https.createServer(options, function (req, res) {
    send(req,
         new url.URL(req.url, 'http://' + req.headers.host).pathname,
         { root: publicdir, cacheControl: false }
    ).pipe(res)
});

const wss = new ws.Server({ server: server, path: '/socket' });

wss.on('connection', function connection(ws) {
    console.log('"x", "y", "z", "gx", "gy", "gz", "ts"');
    ws.on('message', function incoming(message) {
        const data = JSON.parse(message);
        console.log(`${data.x}, ${data.y}, ${data.z}, ${data.gx}, ${data.gy}, ${data.gz}, ${data.ts}`)
    });
});

server.listen(8080, '0.0.0.0', function () {
    console.error(`listening on https://${server.address().address}:8080`);
});
