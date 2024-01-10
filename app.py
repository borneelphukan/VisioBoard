from flask import Flask,request, render_template, jsonify, send_file
import subprocess
import os
import shutil
from mnist.dataset.load_show_mnist import load_mnist

app = Flask(__name__)
python_command = shutil.which("python") or shutil.which("python3")

@app.route('/')
def dashboard():
    return render_template("index.html")

@app.route("/custom_cnn", methods=["POST"])
def rnn_model():
    try:
        model_type = request.form.get("model_type")
        if model_type == "custom_cnn":
            cnn_model = cnn_model()
            return jsonify({"message": "CNN Model loaded successfully."})
        else:
            return jsonify({"error": "Invalid model type."})
    except Exception as e:
        return jsonify({"error": str(e)})
    
# load mnist data
@app.route("/load_dataset", methods=["POST"]) 
def load_dataset():
    try:
        dataset_type = request.form.get("dataset_type")
        if dataset_type == "mnist":
            data = load_mnist()
            return jsonify({"message": "MNIST Dataset loaded."})
        else:
            return jsonify({"error": "Invalid dataset type."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_image')
def get_image():
    image_path = 'static/images/dataset.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/get_image/show_mnist')
def show_mnist():
    show_mnist = subprocess.run(["python", "visualize.py"])
    return show_mnist

# train mnist data
@app.route('/train', methods=['POST'])
def train_model(): 
    if request.method == 'POST':
        current_directory = os.getcwd()
        train_path = os.path.join(current_directory, 'mnist', 'train.py')
        try:
            subprocess.run([python_command, train_path], check=True, cwd=current_directory)
            result_message = "Training completed successfully!"
        except subprocess.CalledProcessError as e:
            result_message = f"Error during training: {e}"
        return render_template('index.html', result_message=result_message)
    
app.run(debug=True)