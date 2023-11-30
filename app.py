from flask import Flask,request, url_for, redirect, render_template

import numpy as np

app = Flask(__name__)



@app.route('/')
def dashboard():
    return render_template("index.html")

app.run(debug=True)
