#import sqlite3
from flask import Flask, render_template
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)
'''
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn
'''
"""
@app.route('/')
def create_app():
  app = Flask(__name__)
  Bootstrap(app)
  return app
  
"""
@app.route('/') # converts the return value into an HTTP response to be displayed by an HTTP client
def index():
    '''conn = get_db_connection() # open a database connection 
    posts = conn.execute('SELECT * FROM posts').fetchall() # execute an SQL query to select all entries from the posts table
    conn.close() # close the database connection
    '''
    return render_template('index.html')#, posts=posts) # can access the blog posts from the index.html template


