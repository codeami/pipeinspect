from flask import Flask, render_template, request, send_from_directory, jsonify
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

@app.route('/test-marker', methods=['POST'])
def test_marker():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    try:
        # Convert uploaded file to numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'})

        # Try to find marker
        processor = PipeImageProcessor(marker_length=1.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = processor._create_red_mask(hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return jsonify({'success': False, 'message': 'No red objects detected in the image'})

        marker = processor._find_marker(contours)
        if marker is None:
            return jsonify({'success': False, 'message': 'Reference marker not found. Please ensure the marker is clearly visible.'})

        # Draw test results on image
        test_image = image.copy()
        cv2.drawContours(test_image, [marker], -1, (0, 255, 0), 2)
        cv2.imwrite('static/test_marker.jpg', test_image)

        return jsonify({
            'success': True, 
            'message': 'Reference marker detected successfully!',
            'test_image': '/static/test_marker.jpg'
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Use the test marker image if it exists
        test_marker_path = 'static/test_marker.jpg'
        if not os.path.exists(test_marker_path):
            return 'Please test marker detection first', 400
            
        image = cv2.imread(test_marker_path)

        if image is None:
            return 'Invalid image format', 400

        # Save input image for reference
        input_path = os.path.join('static', 'input.jpg')
        cv2.imwrite(input_path, image)

        # Process image
        processor = PipeImageProcessor(marker_length=1.0)
        results = processor.process_image(image)
        output_path = display_results(image, results)

        # Prepare measurement data for tooltips
        measurements = []
        for pipe in results['pipe_measurements']:
            # Calculate centroid for tooltip positioning
            M = cv2.moments(pipe['contour'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                measurements.append({
                    'width_mm': pipe['width_mm'],
                    'length_m': pipe['length_m'],
                    'category': pipe['category'],
                    'shape_complexity': pipe['shape_complexity'],
                    'center': {'x': cx, 'y': cy}
                })

        return render_template('result.html', 
                             image_path=output_path,
                             input_path=input_path,
                             measurements=measurements)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f'Error processing image: {str(e)}', 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)