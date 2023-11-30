from flask import Flask,request, url_for, redirect, render_template

import numpy as np

app = Flask(__name__)



@app.route('/')
def homePage():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
