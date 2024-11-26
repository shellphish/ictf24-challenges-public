import logging
import os
import random
import re
import sqlite3
import string

from dotenv import load_dotenv
from flask import Flask, request, render_template_string, make_response, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI

from chatbot import get_answer
from config import DATABASE
from queries import CREATE_TABLE_COOKIES, PROVISION_COOKIES, VALIDATE_ADMIN_COOKIE, VALIDATE_PLAY_TOKEN, \
    CREATE_TABLE_MESSAGES, GET_CHAT_HISTORY, SAVE_USER_MESSAGE, SAVE_SERVER_MESSAGE, GET_ADMIN_COOKIE
from templates import home_template, chat_template, admin_page_template, access_denied_template
from webdriver import load_admin_page

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = None
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
CTF_FLAG = os.getenv("CTF_FLAG")

# Set up Flask-Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)


def init_db():
    if not os.path.exists(DATABASE):
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(CREATE_TABLE_COOKIES)
            conn.execute(CREATE_TABLE_MESSAGES)
            conn.commit()
            logger.info("Database initialized and tables created.")


def log_request_info(route):
    ip_address = request.remote_addr
    user_agent = request.user_agent
    browser = user_agent.browser
    browser_version = user_agent.version
    os_platform = user_agent.platform
    os_version = user_agent.version

    logger.info(
        f"Route: {route}, IP Address: {ip_address}, Browser: {browser} {browser_version}, OS: {os_platform} {os_version}")


@app.route('/')
@limiter.limit("120 per minute")
def index():
    log_request_info(route='/')
    return render_template_string(home_template)


@app.route('/play')
@limiter.limit("60 per minute")
def play_game():
    log_request_info(route='/play')
    play_token = request.cookies.get('play_token')
    logger.info(f"route /play : play_token: {play_token}")

    if not play_token:
        global client
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        # Generate a random admin_cookie and play_token
        admin_cookie = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        play_token = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        # Store the cookies in the SQLite database
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(PROVISION_COOKIES, (admin_cookie, play_token))
            conn.commit()

        logger.info(f"New admin_cookie created: {admin_cookie}, play_token created: {play_token}")

    # Retrieve chat history from the database
    chat_history = []
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute(GET_CHAT_HISTORY, (play_token,))
        chat_history = cursor.fetchall()

    # Set the cookies in the user's browser
    resp = make_response(render_template_string(chat_template, chat_history=chat_history))
    resp.set_cookie('play_token', play_token)
    return resp


@app.route('/admin')
@limiter.limit("30 per minute")
def admin():
    log_request_info(route='/admin')
    admin_cookie = request.cookies.get('admin_cookie')
    play_token = request.cookies.get('play_token')
    logger.info(f"route /admin : admin_cookie: {admin_cookie}, play_token: {play_token}")

    if not admin_cookie or not play_token:
        return render_template_string(access_denied_template)

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute(VALIDATE_ADMIN_COOKIE, (admin_cookie, play_token))
        if cursor.fetchone():
            chat_history = []
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.execute(GET_CHAT_HISTORY, (play_token,))
                chat_history = cursor.fetchall()
            return render_template_string(admin_page_template, FLAG=CTF_FLAG, username="Admin",
                                          chat_history=chat_history)
        else:
            logger.warning("Access denied: Invalid admin_cookie or play_token.")
            return render_template_string(access_denied_template)


@app.route('/send_message', methods=['POST'])
@limiter.limit("60 per minute")
def send_message():
    global client
    message = request.json.get('message')
    play_token = request.cookies.get('play_token')
    logger.info(f"Received message: {message}, play_token: {play_token}")

    if not play_token:
        return jsonify({'error': 'Access denied: No play token provided'}), 403

    # Check word count
    word_count = len(message.split())
    if word_count > 35:
        return jsonify({'response': 'Do you think you smart haha :)'})

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute(VALIDATE_PLAY_TOKEN, (play_token,))
        if cursor.fetchone():
            # Save the user's message
            conn.execute(SAVE_USER_MESSAGE,
                         (play_token, message, 0))
            conn.commit()

            if re.match(r"i\s*need\s*a\s*human", message, re.IGNORECASE):
                # Get the admin cookie from the database associated with the play token
                cursor = conn.execute(GET_ADMIN_COOKIE, (play_token,))
                try:
                    admin_cookie = cursor.fetchone()[0]
                    isLoaded = load_admin_page(admin_cookie=admin_cookie, play_token=play_token)
                    logger.info(f" web driver successfully loaded {isLoaded}")
                except Exception as e:
                    logger.error(f"Error loading web driver: {e}")
                    return jsonify({'response': 'Apologies, No human support available'})
                return jsonify({'response': 'No human support available'})
            try:
                response_message = get_answer(client=client, question=message)
            except Exception as e:
                load_dotenv()
                client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
                response_message = get_answer(client=client, question=message)

            # Save the server's response
            conn.execute(SAVE_SERVER_MESSAGE,
                         (play_token, response_message, 1))
            conn.commit()

            return jsonify({'response': response_message})
        else:
            return jsonify({'error': 'Access denied: Invalid play token'}), 403


if __name__ == '__main__':
    logger.info("Starting server...")
    init_db()
    logger.info("Database initialized.")
    app.run(debug=False, host='0.0.0.0', port=5001)
