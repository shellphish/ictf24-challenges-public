import os
import subprocess as sp
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)

# Initialize the limiter with rate limit rules
limiter = Limiter(
    get_remote_address, app=app, default_limits=["200 per day", "50 per hour"]
)

UPLOAD_FOLDER = "/tmp"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # Set max file size to 1MB

@app.route("/", methods=["GET"])
def index():
    return "Hello! Please upload a file to /upload endpoint", 200

# Route for file upload
@app.route("/upload", methods=["POST"])
@limiter.limit("10 per minute")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        # Generate a unique filename
        unique_filename = f"{uuid.uuid4().hex}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        try:
            file.save(file_path)
            res = run_compute(file_path)
            os.remove(file_path)
            return jsonify(
                {"result": res}
            ), 200
        except RequestEntityTooLarge:
            return jsonify({"error": "File size exceeds 1MB limit"}), 413


def run_compute(file_path):
    # Run the compute script with the uploaded file
    try:
        res = sp.check_output(
            ["su", "compute", "-c", f"python3 /home/challenge/src/compute.py {file_path}"], timeout=10, cwd="/work"
        )
        return res.decode("utf-8").strip()
    except sp.CalledProcessError:
        return "Invalide Matrix"


# Run the app
if __name__ == "__main__":
    app.run(port=32123, host="0.0.0.0")
