chat_template = '''
<!doctype html>
<html>
    <head>
        <title>Chat Page</title>
        <!-- Bootstrap CSS -->
        <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <style>
            body { 
                background-color: black; 
                color: #00ffea; 
                font-family: 'Courier New', Courier, monospace; 
            }
            .chat-container {
                width: 80%;
                margin: auto;
                padding: 20px;
                border: 1px solid #00ffea;
                border-radius: 10px;
                background-color: #111;
            }
            .messages {
                height: 400px;
                overflow-y: scroll;
                border: 1px solid #ff00ff;
                padding: 10px;
                border-radius: 5px;
                background-color: #222;
            }
            .input-container {
                margin-top: 20px;
                display: flex;
                flex-direction: column;
            }
            .input-container input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ff00ff;
                border-radius: 5px;
                background-color: #333;
                color: #00ffea;
            }
            .input-container button {
                padding: 10px;
                border: 2px solid #ff00ff;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #111;
                color: #00ffea;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .input-container button:hover {
                color: #111;
                background-color: #ff00ff;
            }
            h2 {
                color: #ff00ff;
            }
            .word-count {
                color: #ff00ff;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h2>Chat Page</h2>
            <div class="messages" id="messages">
                {% for message, is_response in chat_history %}
                    <div>{{ 'Server: ' if is_response else 'You: ' }}{{ message }}</div>
                {% endfor %}
            </div>
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Type a message...(Type I need a human if you want human support)" oninput="updateWordCount()" onkeypress="checkEnter(event)">
                <span class="word-count" id="wordCount">Word Count: 0</span>
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        <script>
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value;
                const wordCount = message.trim().split(/\s+/).length;
                if (wordCount <= 35) {
                    if (message.trim() !== '') {
                        const messagesDiv = document.getElementById('messages');
                        const newMessage = document.createElement('div');
                        newMessage.textContent = 'You: ' + message;
                        messagesDiv.appendChild(newMessage);

                        const playToken = getCookie('play_token');
                        fetch('/send_message', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Authorization': `Bearer ${playToken}`
                            },
                            body: JSON.stringify({ message: message })
                        })
                        .then(response => response.json())
                        .then(data => {
                            const responseMessage = document.createElement('div');
                            responseMessage.textContent = 'Server: ' + data.response;
                            messagesDiv.appendChild(responseMessage);
                            messagesDiv.scrollTop = messagesDiv.scrollHeight;
                        });

                        input.value = '';
                        updateWordCount();
                    }
                } else {
                    alert('Message cannot exceed 35 words.');
                }
            }

            function getCookie(name) {
                const value = `; ${document.cookie}`;
                const parts = value.split(`; ${name}=`);
                if (parts.length === 2) return parts.pop().split(';').shift();
            }

            function checkEnter(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            function updateWordCount() {
                const input = document.getElementById('messageInput');
                const message = input.value;
                const wordCount = message.trim().split(/\s+/).filter(Boolean).length;
                document.getElementById('wordCount').textContent = 'Word Count: ' + wordCount;
            }
        </script>
    </body>
</html>
'''
