CREATE_TABLE_COOKIES = '''
                CREATE TABLE cookies (
                    admin_cookie TEXT,
                    play_token TEXT
                )
            '''

PROVISION_COOKIES = 'INSERT INTO cookies (admin_cookie, play_token) VALUES (?, ?)'

VALIDATE_ADMIN_COOKIE = 'SELECT * FROM cookies WHERE admin_cookie = ? AND play_token = ?'

VALIDATE_PLAY_TOKEN = 'SELECT * FROM cookies WHERE play_token = ?'

CREATE_TABLE_MESSAGES = '''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    play_token TEXT NOT NULL,
    message TEXT NOT NULL,
    is_response INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
'''

GET_CHAT_HISTORY = "SELECT message, is_response FROM messages WHERE play_token = ? ORDER BY timestamp"

SAVE_USER_MESSAGE = "INSERT INTO messages (play_token, message, is_response) VALUES (?, ?, ?)"

SAVE_SERVER_MESSAGE = "INSERT INTO messages (play_token, message, is_response) VALUES (?, ?, ?)"

GET_ADMIN_COOKIE = 'SELECT admin_cookie FROM cookies WHERE play_token = ?'
