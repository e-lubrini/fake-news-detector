#import sqlite3
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from wtforms import StringField, validators
from flask_wtf import FlaskForm

import os

class URLForm(FlaskForm):
    url = StringField('URL', [validators.DataRequired(message='Field required')])

app = Flask(__name__)
Bootstrap(app)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

@app.route('/') # converts the return value into an HTTP response to be displayed by an HTTP client
def index():
    print('HERE')
    form = URLForm()
    if form.validate_on_submit():
        pass # do something
    return render_template('index.html', form=form)
def popup():
    return render_template('popup.html')