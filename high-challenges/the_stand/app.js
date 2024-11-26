const express = require('express');
const bodyParser = require('body-parser');
const neo4j = require('neo4j-driver');

const app = express();
const port = 5000;

// Neo4j connection setup
const driver = neo4j.driver('bolt://localhost:7687', neo4j.auth.basic('neo4j', 'securepassword'));
const session = driver.session();

app.use(bodyParser.json());

const path = require('path');

// Serve the index.html file
app.use(express.static(path.join(__dirname + "/src")));

// Endpoint to get all people
app.get('/people', async (req, res) => {
    try {
        const result = await session.run('MATCH (p:Person) RETURN p.firstName, p.lastName');
        const people = result.records.map(record => ({
            firstName: record.get('p.firstName'),
            lastName: record.get('p.lastName'),
        }));
        res.json(people);
    } catch (err) {
        res.status(500).send('Error fetching people.');
    }
});

app.get('/friends', async (req, res) => {
    const { firstName, lastName, debug } = req.query;

    // Construct the query
    const query = `MATCH (p:Person {firstName: '${firstName}', lastName: '${lastName}'}) OPTIONAL MATCH (p)-[:FRIEND]->(f) RETURN p.tag AS personTag, f.firstName AS firstName, f.lastName AS lastName, f.tag AS friendTag
    `;

    try {
        const result = await session.run(query);

        // Check if the person exists
        if (result.records.length === 0) {
            const response = { message: "This person does not exist." };
            if (debug) response.query = query;
            res.json(response);
            return;
        }

        // Filter out records with no friends
        const friends = result.records.filter(record => record.get('firstName') !== null);

        if (friends.length === 0) {
            // If no friends, prepare the "no friends" response
            const response = { message: "This person has no friends (sad!)" };
            if (debug) response.query = query;
            res.json(response);
        } else {
            // If friends exist, prepare the full response
            const personTag = result.records[0].get('personTag');
            const response = {
                message: `${firstName} ${lastName} says: ${personTag}`,
                friends: friends.map(record => {
                    const friendFirstName = record.get('firstName');
                    const friendLastName = record.get('lastName');
                    const friendTag = record.get('friendTag');
                    return `${friendFirstName} ${friendLastName} says: ${friendTag}`;
                })
            };

            // Include the query if debug is enabled
            if (debug) response.query = query;
            res.json(response);
        }
    } catch (err) {
        console.error('Error fetching friends:', err);
        res.status(500).json({ error: 'Error fetching friends.', query: debug === "True" ? query : undefined });
    }
});

// Start the server
app.listen(port, "0.0.0.0", () => {
    console.log(`App running at http://0.0.0.0:${port}`);
});

// Cleanup
process.on('SIGINT', async () => {
    await session.close();
    await driver.close();
    process.exit(0);
});
