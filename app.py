from flask import Flask,request, url_for, redirect, render_template, jsonify, send_file
from dopamine.datasets.figure8 import Figure8Dataset, load_figure8_dataset
from dopamine.datasets.lorenz import LorenzAttractorDataset, load_lorenz_attractor_dataset
from dopamine.datasets.rossler import RosslerAttractorDataset, load_rossler_attractor_dataset
from dopamine.utils import load_dataset_image
from dopamine.optimizer import Dopamine
from dopamine.models.rnn import RNNModel
import subprocess
import os
import shutil
import numpy as np

app = Flask(__name__)

python_command = shutil.which("python") or shutil.which("python3")

@app.route('/')
def dashboard():
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    dopamine_optimizer = Dopamine()
    return jsonify({"message": "Optimization with Dopamine completed."})

@app.route("/rnn_model", methods=["POST"])
def rnn_model():
    try:
        model_type = request.form.get("model_type")

        if model_type == "rnn_model":
            input_size, output_size, hidden_dim = 10, 5, 20
            rnn_model = RNNModel(input_size, output_size, hidden_dim)
            return jsonify({"message": "RNN Model loaded successfully."})
        else:
            return jsonify({"error": "Invalid model type."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_image')
def get_image():
    image_path = 'static/images/dataset.png'
    return send_file(image_path, mimetype='image/png')
    
@app.route('/get_image/load_figure8_image')
def load_figure8_image():
    image_path = load_dataset_image('figure8.py')
    return jsonify({'image_path': image_path})

@app.route('/get_image/load_lorenz_image')
def load_lorenz_image():
    return load_dataset_image('lorenz.py')

@app.route('/get_image/load_rossler_image')
def load_rossler_image():
    return load_dataset_image('rossler.py')


@app.route("/load_dataset", methods=["POST"])
def load_dataset():
    try:
        dataset_type = request.form.get("dataset_type")

        if dataset_type == "figure8":
            # Dummy values for Figure8Dataset instantiation
            sequence_length = 50
            target_length = 1

            # Load the Figure8Dataset
            data = load_figure8_dataset()
            figure8_dataset = Figure8Dataset(data, sequence_length, target_length)

            # Dummy response, you can return any information you want
            return jsonify({"message": "Figure 8 Dataset loaded successfully."})

        elif dataset_type == "lorenz":
            # Dummy values for LorenzAttractorDataset instantiation
            sequence_length = 499
            target_length = 1

            # Load the LorenzAttractorDataset
            data = load_lorenz_attractor_dataset(data_length=500)
            lorenz_dataset = LorenzAttractorDataset(data, sequence_length, target_length)

            # Dummy response, you can return any information you want
            return jsonify({"message": "Lorenz Dataset loaded successfully."})
        
        elif dataset_type == "rossler":
            # Load the RosslerAttractorDataset
            sequence_length = 499
            target_length = 1
            data = load_rossler_attractor_dataset(data_length=500)
            rossler_dataset = RosslerAttractorDataset(data, sequence_length, target_length)

            # Dummy response, you can return any information you want
            return jsonify({"message": "Rossler Dataset loaded successfully."})

        else:
            return jsonify({"error": "Invalid dataset type."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/train', methods=['POST'])
def train_model():
    if request.method == 'POST':
        # Get the current directory
        current_directory = os.getcwd()
        
        # Set the path to the train_wp.py script
        train_wp_path = os.path.join(current_directory, 'dopamine', 'train_wp.py')

        # Execute the train_wp.py script using subprocess
        try:
            subprocess.run([python_command, train_wp_path], check=True, cwd=current_directory)
            result_message = "Training completed successfully!"
        except subprocess.CalledProcessError as e:
            result_message = f"Error during training: {e}"

        return render_template('index.html', result_message=result_message)




app.run(debug=True)
