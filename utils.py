import cv2
import numpy as np
from typing import Dict
import os

def display_results(image: np.ndarray, results: Dict) -> str:
    """Process the image with measurements and save results"""
    # Create a copy of the image for drawing
    output = image.copy()

    # Draw marker contour
    cv2.drawContours(output, [results['marker_contour']], -1, (0, 255, 0), 2)

    # Draw pipe measurements
    for pipe in results['pipe_measurements']:
        contour = pipe['contour']
        length = pipe['length']

        # Draw the contour
        cv2.drawContours(output, [contour], -1, (0, 255, 255), 2)

        # Calculate centroid for text placement
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw measurement text
            text = f"{length}m"
            cv2.putText(output, text, (cx-20, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Print results
    print("\nPipe Measurements:")
    for i, pipe in enumerate(results['pipe_measurements'], 1):
        print(f"Pipe {i}: {pipe['length']} meters")

    # Save the image
    if not os.path.exists('static'):
        os.makedirs('static')
    output_path = 'static/processed_image.jpg'
    cv2.imwrite(output_path, output)

    return output_path