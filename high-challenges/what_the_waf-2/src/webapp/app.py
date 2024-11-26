
import  ldap
import base64
import hashlib
import subprocess
import os
import uuid

from flask import Flask, request, redirect, url_for, render_template_string, render_template, make_response

# rate limiter 
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from datetime import datetime
from waf import llm_protect


def default_error_responder(request_limit):
    return make_response(
        render_template("dos_protection.html", request_limit=request_limit),
        429
    )

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    on_breach=default_error_responder
)


# Configuring LDAP
LDAP_SERVER = 'ldap://localhost'
LDAP_BASE_DN = 'dc=example,dc=com'
LDAP_USER_DN = "ou=users," + LDAP_BASE_DN

CHECK_HEADERS = ['User-Agent', 'Referer', 'Host', 'Origin']

def calculate_md5(input_string):
    # Encode the string to bytes
    encoded_string = input_string.encode()
    # Create an MD5 hash object
    md5_hash = hashlib.md5()
    # Update the hash object with the encoded string
    md5_hash.update(encoded_string)
    # Get the hexadecimal representation of the hash
    hash_hex = md5_hash.hexdigest()
    return hash_hex

def get_raw_http_request(request):
    HTTP_RAW = ''
    # extract method
    HTTP_RAW += f'{request.method} {request.path}\n'
    for k,v in request.headers.items():
        if k in CHECK_HEADERS:
            HTTP_RAW += f'{k}: {v}\n'

    HTTP_RAW += '\n'

    # put all the forms
    for k,v in request.form.items():
        HTTP_RAW += f'{k}={v}&'
    # remove last &
    HTTP_RAW = HTTP_RAW[:-1]
    return HTTP_RAW


@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))

@app.route('/robots.txt')
def robot():
    # 🧐: CLUE, /dev-logs discloses the log of the webapp
    # so they can have feedback on the LDAP injection and LLM bypass.
    return 'User-agent: GPTBot\nDisallow: /dev-logs'

@app.route('/dev-logs')
def debug_logs():
    user_uuid = request.cookies.get('user_uuid')

    if not user_uuid:
        user_uuid = str(uuid.uuid4())

    data = ''

    # Add the generic logs (this is the hint about ldap etc...)
    log_generic = f'logs/app_debug'
    if os.path.exists(log_generic):
        with open(log_generic, 'r') as f:
            data = f.read().replace('\n', '<br>')

    # Add the uuid
    data += f'Logs for user {user_uuid}<br><br>'
    
    # Now add per-user logs
    log_file_path = f'logs/app_debug-{user_uuid}'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            data += f.read().replace('\n', '<br>')

    response = make_response(data)
    response.set_cookie('user_uuid', user_uuid)

    return data

def log_stuff(user_uuid, log_message):
    now = datetime.now()
    log_file_path = f'logs/app_debug-{user_uuid}'
    with open(log_file_path, 'a') as f:
        f.write(now.strftime('%Y-%m-%d %H:%M:%S') + " | ")
        f.write(log_message)
        f.write('\n')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("30/minute;2/second")
def login():
    # Do we have a user id? If not, create one
    user_uuid = request.cookies.get('user_uuid')
    if not user_uuid:
        user_uuid = str(uuid.uuid4())

    raw_http_request = get_raw_http_request(request)
    
    # Cap the raw HTTP requesto to 2000 bytes
    if len(raw_http_request) > 2000:
        response = make_response("Request too long...")
        response.set_cookie('user_uuid', user_uuid)
        return response

    request_should_pass = llm_protect(user_uuid, raw_http_request)
    if not request_should_pass:
        response = make_response(render_template('waf.html'))
        response.set_cookie('user_uuid', user_uuid)
        return response

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "*":
            print(f'Static analysis detected a potential LDAP injection attack')
            # set the cookie too
            response = make_response("Invalid username *")
            response.set_cookie('user_uuid', user_uuid)
            return response

        try:
            conn = ldap.initialize(LDAP_SERVER)
            conn.set_option(ldap.OPT_REFERRALS, 0)
            conn.protocol_version = ldap.VERSION3
            conn.simple_bind_s("cn=admin,dc=example,dc=com", "megalol")

            search_filter ="(& (uid=" + username + ")" + "(userPassword=" + password + "))"

            print(f'search_filter: {search_filter}')
            result = conn.search_s(LDAP_USER_DN,ldap.SCOPE_SUBTREE, search_filter)
            print(f'search_results: {result}')

            log_stuff(user_uuid, f'User: {username} | Password: {password} | Search Results: {result}\n')

            # if we have results, bind as that user!
            if result:
                user_dn = result[0][0]
                try:
                    print(f'Attempting binding with user_dn {user_dn} | password: {password}')
                    conn.simple_bind_s(user_dn, password)
                    print(f'User {username} found, binding...')
                    if username == "tonystark":
                        message = "Great job, you found the flag :)"
                        eth = "10"
                        btc = "6"
                        pepe = "123"
                    else:
                        eth = "0"
                        btc = "0"
                        pepe = "0"
                        message = "You are CTF-close, flag is not here, try another user ;)"
                    response = make_response(render_template('user.html', username=username, message=message, eth=eth, btc=btc, pepe=pepe))
                    response.set_cookie('user_uuid', user_uuid)
                    return response

                except ldap.INVALID_CREDENTIALS:
                    response = make_response("Invalid credentials for user " + username)
                    response.set_cookie('user_uuid', user_uuid)
                    return response
            else:
                response = make_response(f'User {username} not found')
                response.set_cookie('user_uuid', user_uuid)
                return response

        except ldap.LDAPError as e:
            log_stuff(user_uuid, f'LDAP error: {str(e)}')
            response = make_response(f'Fatal error during user login: {str(e)}')
            response.set_cookie('user_uuid', user_uuid)
            return response

    response = make_response(render_template('login.html'))
    assert user_uuid
    response.set_cookie('user_uuid', user_uuid)

    return response

if __name__ == '__main__':
    # Write starting message to the debug log
    # Remove the file first if it exits
    if os.path.exists('logs/app_debug'):
        os.remove('logs/app_debug')

    # create the file now
    os.system('touch logs/app_debug')

    with open('logs/app_debug', 'w') as f:
        f.write("==============================================\n")
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'Starting the application at {start_time}\n\n')
        # get ldap version, catch the output and write it
        cmd = ["/usr/sbin/slapd", "-VV"]
        pid = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = pid.communicate()
        ldap_version = stderr.decode('utf-8')
        f.write(f'BACKEND DATABASE: {ldap_version}')
        f.write(f'WAF PROTECTION ACTIVATED: LLM-WAF (GPT4o, max_context:4096) up and running!\n')
        f.write("==============================================\n")
        f.flush()

    app.run(host="0.0.0.0", port=9999)
