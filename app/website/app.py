#import sqlite3
from flask import Flask, render_template
from flask_bootstrap import Bootstrap


app = Flask(__name__)
Bootstrap(app)

@app.route('/') # converts the return value into an HTTP response to be displayed by an HTTP client
def index():
    return render_template('index.html')