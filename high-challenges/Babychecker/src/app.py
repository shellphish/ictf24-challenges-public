from flask import Flask, render_template, session, request, redirect, flash
import os
import requests
from bot import checker
app = Flask(__name__)

flag = os.environ.get('FLAG')

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/send', methods=['POST'])
def visit_website():
    link = request.form.get('data')
    checkres = checker(link)
    if checkres["result"] == True:
        return {"res": "hacker","reason": checkres["reason"]}
    elif checkres["result"] == False:
        try:
            res = requests.get(checkres["url"], allow_redirects=False, timeout=20).text
            
            return {"res":res, "reason": checkres["reason"]}
        except Exception as e:
            return {"res":"Error", "reason": str(e)}
    else:
        return {"res": "Error", "reason": "Error"}


@app.route('/flag', methods=['GET'])
def getflag():
    if request.remote_addr == '127.0.0.1':
        return render_template('result.html', msg=flag)
    else:
        return render_template('result.html', msg="hacker")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
