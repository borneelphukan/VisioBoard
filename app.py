from flask import Flask,request, url_for, redirect, render_template, jsonify
from dopamine.optimizer import Dopamine

import numpy as np

app = Flask(__name__)



@app.route('/')
def dashboard():
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    # Instantiate the Dopamine class and perform optimization
    dopamine_optimizer = Dopamine()
    # Perform optimization steps here using the Dopamine instance

    # Dummy response, you can return any information you want
    return jsonify({"message": "Optimization with Dopamine completed."})


app.run(debug=True)
