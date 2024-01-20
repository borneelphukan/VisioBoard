from flask import Flask, render_template, jsonify, request, send_file
import subprocess
import shutil
from backend.datasets.load_fashion_mnist import load_fashion_mnist
from backend.datasets.load_mnist import load_mnist

app = Flask(__name__)
python_command = shutil.which("python") or shutil.which("python3")

#loads the UI
@app.route('/')
def dashboard():
    return render_template("index.html")

#loads model on selecting Model Dropdown
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

#loads datasets/load_mnist
@app.route('/load_mnist', methods=['POST'])
def load_mnist_data():
    try:
        result = load_mnist()
        return {"status": "success", "message": "MNIST Dataset loaded successfully", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
#loads datasets/load_fashion_mnist
@app.route('/load_fashion_mnist', methods=['POST'])
def load_fashion_mnist_data():
    try:
        result = load_fashion_mnist()
        return {"status": "success", "message": "Fashion MNIST Dataset loaded successfully", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

#fetch dataset plot
@app.route('/get_image')
def get_image():
    image_path = 'static/images/dataset.png'
    return send_file(image_path, mimetype='image/png')

#fetch training plot
@app.route('/get_tplot')
def get_training_image():
    image_path = 'static/images/cnn_1.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Use subprocess to execute the train.py script
        subprocess.run([python_command, 'backend/train.py'], check=True)
        
        # Set the result message
        result_message = "Model trained and saved as cnn_1_weights.h5"
    except Exception as e:
        # Handle exceptions if the training fails
        result_message = f"Error during training: {str(e)}"

    # Render your template with the result message
    return render_template('index.html', result_message=result_message)

app.run(debug=True)