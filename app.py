from flask import Flask, render_template, jsonify, request, send_file
import subprocess
import shutil
from backend.datasets.load_fashion_mnist import load_fashion_mnist
from backend.datasets.load_mnist import load_mnist

app = Flask(__name__)
python_command = shutil.which("python") or shutil.which("python3")

@app.route('/')
def dashboard():
    return render_template("index.html")

@app.route('/load_model', methods=['POST'])
def load_model():
    selected_model = request.form.get('selected_model')
    if selected_model == 'cnn_1':
        try:
            subprocess.run([python_command, 'backend/models/cnn_1.py'])
            return jsonify({'success': 'CNN_1 loaded successfully'})
        except Exception as e:
            return jsonify({'error': 'The model outputted error. Check the model code and update the errors'})
    else:
        return jsonify({'error': 'Invalid model selected'})

@app.route('/load_mnist', methods=['POST'])
def load_mnist_data():
    try:
        # Execute the function
        result = load_mnist()
        return {"status": "success", "message": "MNIST Dataset loaded successfully", "data": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.route('/load_fashion_mnist', methods=['POST'])
def load_fashion_mnist_data():
    try:
        # Execute the function
        result = load_fashion_mnist()
        return {"status": "success", "message": "Fashion MNIST Dataset loaded successfully", "data": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/get_image')
def get_image():
    # Specify the path to your dataset.png
    image_path = 'static/images/dataset.png'
    return send_file(image_path, mimetype='image/png')

app.run(debug=True)