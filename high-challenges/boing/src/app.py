import threading
import logging
import hashlib
import traceback
import subprocess
import secrets
from flask import Flask, request, render_template, redirect, url_for, session, g
import time
import os
import sqlite3

TMPDIR = '/tmp/boing'
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'pages')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
DB_PATH = '/tmp/users.db'

app = Flask(__name__)
app.template_folder = TEMPLATES_DIR
app.static_folder = STATIC_DIR

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
    return db


@app.route('/')
def index():
    dat = {}
    if 'username' in session:
        dat['username'] = session['username']
        unprocessed_files = []
        user_files_dir = os.path.join(TMPDIR, str(session['user_id']))
        if os.path.exists(user_files_dir):
            for fname in os.listdir(user_files_dir):
                if is_jpg_ext(fname):
                    unprocessed_files.append(fname)
        processed_files = []
        user_static_dir = os.path.join(STATIC_DIR, str(session['user_id']))
        if os.path.exists(user_static_dir):
            for fname in os.listdir(user_static_dir):
                if is_jpg_ext(fname):
                    processed_files.append((fname, get_score(os.path.join(user_static_dir, fname))))
        dat['unprocessed_files'] = set(unprocessed_files) - set(processed_files)
        dat['processed_files'] = processed_files

    dat['flag'] = os.environ.get('FLAG', 'ictf{this_is_a_fake_flag}')
    return render_template('index.html', **dat)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        db = get_db()
        c = db.cursor()
        c.execute('SELECT id FROM users WHERE username = ? AND password_sha = ?', (username, password_hash))
        maybe_rowid = c.fetchone()
        if maybe_rowid:
            user_id = str(maybe_rowid[0])
            session['username'] = username
            session['user_id'] = user_id
            app.logger.info(f'User {username}/{user_id} logged in')
            return redirect(url_for('index'))
        else:
            app.logger.info(f'Failed login attempt for user {username}')
            return render_template('login.html', error='Invalid username or password'), 403

    else:
        return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        db = get_db()
        c = db.cursor()
        try:
            user_id = secrets.token_hex(16)
            c.execute('INSERT INTO users (id, username, password_sha) VALUES (?, ?, ?)', (user_id, username, password_hash))
            app.logger.info(f'Created user {username}')
            db.commit()
            return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already taken'), 403
    else:
        return render_template('register.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', username=session.get('username'), error='No file provided'), 400

        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', username=session.get('username'), error='No file provided'), 400

        if not is_jpg_ext(file.filename):
            return render_template('upload.html', username=session.get('username'), error='Invalid file extension'), 400

        user_files_dir = os.path.join(TMPDIR, str(session['user_id']))
        if not os.path.exists(user_files_dir):
            os.mkdir(user_files_dir)
        ext = os.path.splitext(file.filename)[-1]
        gen_file_name = secrets.token_hex(16) + ext
        fpath = os.path.join(user_files_dir, gen_file_name)
        file.save(fpath)

        try:
            compute_metadata(fpath)
            return render_template('upload.html', username=session.get('username'), success='File uploaded successfully', file_name=gen_file_name)
        except:
            print('Error processing file')
            traceback.print_exc()
            os.unlink(fpath)
            return render_template('upload.html', username=session.get('username'), error=f'Error processing file {gen_file_name}'), 400
    else:
        return render_template('upload.html', username=session.get('username'))


def compute_metadata(fpath):
    output_file = fpath + '.meta'
    cmd = ['python3', 'compute.py', fpath, output_file]
    subprocess.run(cmd, timeout=1)


@app.route('/process', methods=['GET', 'POST'])
def process_file():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_files_dir = os.path.join(TMPDIR, str(session['user_id']))

    if request.method == 'POST':
        file_name = request.form['file_name'] + '.meta'
        # Ensure that the file is in the TMPDIR
        abs_file_path = os.path.abspath(os.path.join(user_files_dir, file_name))
        if not abs_file_path.startswith(user_files_dir):
            return render_template('process.html', username=session.get('username'), error='Invalid file path'), 400

        if not os.path.exists(abs_file_path):
            return render_template('process.html', username=session.get('username'), error='File not found'), 400

        # Find the filename of the original file
        with open(abs_file_path, 'r') as f:
            orig_file_name = None
            for line in f:
                if line.startswith('Filename: '):
                    orig_file_name = line.split(': ')[1].strip()
            if not orig_file_name:
                return render_template('process.html', username=session.get('username'), error='Invalid metadata file'), 400
        
        original_file_path = os.path.join(user_files_dir, orig_file_name)
        if not os.path.exists(original_file_path):
            return render_template('process.html', username=session.get('username'), error='Original file not found'), 400

        # Expose the file in the static directory
        user_static_dir = os.path.join(STATIC_DIR, str(session['user_id']))
        if not os.path.exists(user_static_dir):
            os.makedirs(user_static_dir)
        
        new_file_path = os.path.join(user_static_dir, os.path.basename(orig_file_name))
        os.symlink(original_file_path, new_file_path)
        new_meta_file_path = os.path.join(user_static_dir, os.path.basename(orig_file_name) + '.meta')
        os.symlink(abs_file_path, new_meta_file_path)

        app.logger.info(f'User {session["username"]} exposed {original_file_path} as {new_file_path}')
        return redirect(url_for('index'))
    else:
        return render_template('process.html', username=session.get('username'))


@app.route('/get/<file_name>')
def get_file(file_name):
    if 'username' not in session:
        return redirect(url_for('login'))

    if not is_jpg_ext(file_name) and not file_name.endswith('.meta'):
        return 'Invalid file extension', 400

    user_static_dir = os.path.join(STATIC_DIR, str(session['user_id']))
    fpath = os.path.join(user_static_dir, file_name)
    if not os.path.exists(fpath):
        return 'File not found', 404

    return app.send_static_file(os.path.join(str(session['user_id']), file_name))


def cleanup_loop():
    while True:
        print('Cleaning up')
        now = time.time()
        for fname in os.listdir(TMPDIR):
            # Delete any files older than 30 minutes
            fpath = os.path.join(TMPDIR, fname)
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > 30 * 60:
                os.unlink(fpath)
        
        time.sleep(60 * 5)


def is_jpg_ext(filename):
    return os.path.splitext(filename)[-1].lower() in ['.jpg', '.jpeg']


def get_score(filename):
    try:
        with open(filename + '.meta', 'r') as f:
            for line in f:
                if line.startswith('Score: '):
                    return float(line.split(': ')[1])
    except:
        pass

    # default score is low :(
    return 0.0


if __name__ == '__main__':

    if not os.path.exists(TMPDIR):
        os.makedirs(TMPDIR)
    
    # Start the cleanup loop
    t = threading.Thread(target=cleanup_loop, daemon=True)
    t.start()

    FLAG = os.environ.get('FLAG', 'ictf{this_is_a_fake_flag}')

    # Set up the database
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (id TEXT, username TEXT UNIQUE, password_sha TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS users (id TEXT, username TEXT UNIQUE, password_sha TEXT)')
    c.execute('INSERT INTO users VALUES ("flagflagflagflag", "flag", ?)', (FLAG,))
    db.commit()
    db.close()

    session_key = os.urandom(16)
    app.secret_key = session_key

    # set log level
    app.logger.setLevel(logging.INFO)

    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024  # 100 KB

    app.run(host='0.0.0.0', port=5000)

