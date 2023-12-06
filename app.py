from flask import Flask,request, url_for, redirect, render_template, jsonify
from dopamine.datasets.figure8 import Figure8Dataset, load_figure8_dataset
from dopamine.datasets.lorenz import LorenzAttractorDataset, load_lorenz_attractor_dataset
from dopamine.datasets.rossler import RosslerAttractorDataset, load_rossler_attractor_dataset
from dopamine.optimizer import Dopamine
from dopamine.models.rnn import RNNModel
import subprocess
import os
import uuid

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

@app.route("/rnn_model", methods=["POST"])
def rnn_model():
    try:
        model_type = request.form.get("model_type")

        if model_type == "rnn_model":
            # Dummy values for input_size, output_size, and hidden_dim
            input_size, output_size, hidden_dim = 10, 5, 20
            
            # Instantiate the RNNModel class
            rnn_model = RNNModel(input_size, output_size, hidden_dim)
            # Perform any additional model loading steps if needed

            # Dummy response, you can return any information you want
            return jsonify({"message": "RNN Model loaded successfully."})
        else:
            return jsonify({"error": "Invalid model type."})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/load_figure8_image')
def load_figure8_image():
    image_path = load_dataset_image('figure8.py')
    return jsonify({'image_path': image_path})

@app.route('/load_lorenz_image')
def load_lorenz_image():
    return load_dataset_image('lorenz.py')

@app.route('/load_rossler_image')
def load_rossler_image():
    return load_dataset_image('rossler.py')
    
def load_dataset_image(script_filename):
    try:
        # Generate a unique filename for the image
        unique_filename = str(uuid.uuid4()) + '.png'
        image_path = os.path.join('static', 'images', unique_filename)

        # Run the specified script to generate the image
        subprocess.run(['python', f'dopamine/datasets/{script_filename}', image_path])

        # Return the path to the generated image without jsonify
        return image_path
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': f'Failed to generate image for {script_filename}'})


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
            subprocess.run(['python', train_wp_path], check=True, cwd=current_directory)
            return "Training completed successfully!"
        except subprocess.CalledProcessError as e:
            return f"Error during training: {e}"

app.run(debug=True)
