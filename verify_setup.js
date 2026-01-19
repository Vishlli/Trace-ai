const http = require('http');

const data = JSON.stringify({
    event_type: 'Verification Test',
    details: {
        test: true,
        timestamp: new Date().toISOString()
    }
});

const options = {
    hostname: 'localhost',
    port: 3000,
    path: '/api/data',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
    }
};

console.log('Testing POST /api/data...');

const req = http.request(options, (res) => {
    console.log(`STATUS: ${res.statusCode}`);

    let body = '';
    res.on('data', (chunk) => {
        body += chunk;
    });

    res.on('end', () => {
        console.log('Response Body:', body);
        if (res.statusCode === 200) {
            console.log('Verfication POST successful.');
        } else {
            console.error('Verification POST failed.');
            process.exit(1);
        }
    });
});

req.on('error', (e) => {
    console.error(`problem with request: ${e.message}`);
    process.exit(1);
});

req.write(data);
req.end();
