from flask import Flask, render_template, request, send_from_directory
import cv2
from image_processor import PipeImageProcessor
from utils import display_results
import os
import numpy as np

app = Flask(__name__)

# Ensure upload directory exists
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    try:
        # Convert uploaded file to numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return 'Invalid image format', 400

        # Save input image for reference
        input_path = os.path.join('static', 'input.jpg')
        cv2.imwrite(input_path, image)

        # Process image
        processor = PipeImageProcessor(marker_length=1.0)
        results = processor.process_image(image)
        output_path = display_results(image, results)

        return render_template('result.html', 
                             image_path=output_path,
                             input_path=input_path)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f'Error processing image: {str(e)}', 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)