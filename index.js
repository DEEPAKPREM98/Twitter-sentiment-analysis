
const express = require('express')
const app = express()
const path = require('path');
const hostname = '127.0.0.1';
const port = 3000;
const fs = require('fs')
app.use(express.urlencoded());
const spawn = require('child_process').spawn;

app.use(express.static(path.join(__dirname, "/static")));

app.get('/', function (req, res) {
    res.sendFile(path.join(__dirname, '/indexi.html'));

});

app.get('/services', function (req, res) {
    res.sendFile(path.join(__dirname, '/services.html'));

});

app.get('/about', function (req, res) {
    res.sendFile(path.join(__dirname, '/about.html'));

});

app.get('/contact', function (req, res) {
    res.sendFile(path.join(__dirname, '/contact.html'));

});

app.get('/h', function (req, res) {

    function chec() {
        res.sendFile(path.join(__dirname, '/h.html'));
    }

    function tyi() {

        fs.readFile('output.txt', (err, data) => {
            if (err) throw err;
            t = (data.toString());
            if (t == 5) {

                clearTimeout(k)
                chec()
            }
        })
    }

    var k = setInterval(tyi, 1000)
});

app.post('/', function (req, res) {
    var o = JSON.stringify(req.body)
    res.sendFile(path.join(__dirname, '/services.html'));
    fs.writeFileSync('b.json', o);
    fs.writeFileSync('output.txt', '1')
    fs.writeFileSync('result.txt', ' ')
    fs.appendFileSync('userdata.json', o);
});

function run() {

    var los = []
    fs.readFile("b.json", function (err, data) {
        // Converting to JSON
        const users = JSON.parse(data);
        u = users.name1
        v = users.name2
        w = users.name3
        x = users.name4
        y = users.name5
        z = users.name6
        lisw = ['hello.py', u, v, w, x, y, z]
        for (Element of lisw) {
            if (Element == null ||
                Element == undefined ||
                Element.length == 0) {
                console.log("deep")
            }
            else {
                los.push(Element)
            }
        }

        j = los.length
        fs.writeFileSync('put.txt', j.toString())


        const scriptExecution = spawn("python", los);

        var uint8arrayToString = function (data) {
            return String.fromCharCode.apply(null, data);
        };

        // Handle normal output
        scriptExecution.stdout.on('data', (data) => {
            console.log(uint8arrayToString(data));
            fs.appendFileSync('result.txt', uint8arrayToString(data))
        });

        // Handle error output
        scriptExecution.stderr.on('data', (data) => {
            // As said before, convert the Uint8Array to a readable string.
            console.log(uint8arrayToString(data));

        });

        scriptExecution.on('exit', (code) => {
            console.log("Process quit with code : " + code);
            fs.writeFileSync('output.txt', '5')

            var v = setInterval(secondcheck, 500);
            function secondcheck() {
                fs.readFile('output.txt', (err, data) => {
                    if (err) throw err;
                    t = (data.toString());
                    if (t == 1) {
                        console.log("true")
                        clearTimeout(v)
                        run()
                    }
                })
            }

        });
    })
}

function check() {
    fs.readFile('output.txt', (err, data) => {
        if (err) throw err;
        t = (data.toString());
        if (t == 1) {
            console.log("true")
            clearTimeout(u)
            run()
        }
    })
}

var u = setInterval(check, 500)

app.listen(port, hostname, () => {
    console.log(`Server running at http://${hostname}:${port}/`);
});
