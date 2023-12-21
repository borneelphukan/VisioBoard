import os
import shutil
import subprocess
import uuid

from flask import jsonify

python_command = shutil.which("python") or shutil.which("python3")

def load_dataset_image(script_filename):
    try:
        # Generate a unique filename for the image
        unique_filename = str(uuid.uuid4()) + '.png'
        image_path = os.path.join('static', 'images', unique_filename)

        # Run the specified script to generate the image
        subprocess.run([python_command, f'dopamine/datasets/{script_filename}', image_path])

        # Return the path to the generated image without jsonify
        return image_path
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': f'Failed to generate image for {script_filename}'})
