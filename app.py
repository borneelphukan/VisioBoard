from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    return

if __name__ == '__main__':
    app.run(debug=True)