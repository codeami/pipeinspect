from flask import Flask, render_template, request, send_from_directory
import cv2
from image_processor import PipeImageProcessor
from utils import display_results
import os

app = Flask(__name__)

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

    # Save uploaded file
    input_path = 'static/input.jpg'
    file.save(input_path)
    
    # Process image
    image = cv2.imread(input_path)
    if image is None:
        return 'Could not read the image', 400

    processor = PipeImageProcessor(marker_length=1.0)
    try:
        results = processor.process_image(image)
        output_path = display_results(image, results)
        return render_template('result.html', image_path=output_path)
    except Exception as e:
        return str(e), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000)
