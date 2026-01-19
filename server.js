const express = require('express');
const { Client } = require('pg');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());

// Configuration
const dbConfig = {
    user: 'postgres',
    host: 'localhost',
    database: 'traceai_db',
    password: 'abhi',
    port: 5432,
};

const postgresConfig = {
    ...dbConfig,
    database: 'postgres', // Connect to default DB to check/create target DB
};

async function setupDatabase() {
    // 1. Create Database if not exists
    const pgClient = new Client(postgresConfig);
    try {
        await pgClient.connect();
        const res = await pgClient.query(`SELECT 1 FROM pg_database WHERE datname = '${dbConfig.database}'`);
        if (res.rowCount === 0) {
            console.log(`Database ${dbConfig.database} not found. Creating...`);
            await pgClient.query(`CREATE DATABASE "${dbConfig.database}"`);
            console.log(`Database ${dbConfig.database} created.`);
        } else {
            console.log(`Database ${dbConfig.database} exists.`);
        }
    } catch (err) {
        console.error('Error checking/creating database:', err);
    } finally {
        await pgClient.end();
    }

    // 2. Create Table
    const dbClient = new Client(dbConfig);
    try {
        await dbClient.connect();
        console.log(`Connected to ${dbConfig.database}`);
        
        const createTableQuery = `
            CREATE TABLE IF NOT EXISTS realtime_data (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(255) NOT NULL,
                details JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `;
        await dbClient.query(createTableQuery);
        console.log('Table "realtime_data" ensured.');
        
        return dbClient; // Return connected client for query usage
    } catch (err) {
        console.error('Error setup table:', err);
    }
}

// Global DB Client setup
let dbClient;
setupDatabase().then(client => {
    dbClient = client;
});

// API Routes
app.post('/api/data', async (req, res) => {
    const { event_type, details } = req.body;
    
    if (!dbClient) {
        return res.status(500).json({ error: 'Database not initialized yet' });
    }

    try {
        const query = 'INSERT INTO realtime_data (event_type, details, created_at) VALUES ($1, $2, NOW()) RETURNING *';
        const values = [event_type, JSON.stringify(details || {})];
        const result = await dbClient.query(query, values);
        console.log('Data saved:', result.rows[0]);
        res.json({ success: true, data: result.rows[0] });
    } catch (err) {
        console.error('Error saving data:', err);
        res.status(500).json({ error: 'Failed to save data' });
    }
});

app.get('/api/data', async (req, res) => {
     if (!dbClient) {
        return res.status(500).json({ error: 'Database not initialized yet' });
    }
    try {
        const result = await dbClient.query('SELECT * FROM realtime_data ORDER BY created_at DESC LIMIT 50');
        res.json(result.rows);
    } catch (err) {
        console.error('Error fetching data:', err);
        res.status(500).json({ error: 'Failed to fetch data' });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
