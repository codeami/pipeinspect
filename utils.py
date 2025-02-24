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

    # Colors for different categories
    category_colors = {
        'small': (0, 255, 255),    # Yellow
        'medium': (0, 165, 255),   # Orange
        'large': (0, 0, 255),      # Red
        'unknown': (128, 128, 128) # Gray
    }

    # Draw pipe measurements
    for pipe in results['pipe_measurements']:
        contour = pipe['contour']
        length = pipe['length_m']
        width = pipe['width_mm']
        category = pipe['category']

        # Draw the contour with category color
        color = category_colors.get(category, (255, 255, 255))
        cv2.drawContours(output, [contour], -1, color, 2)

        # Calculate centroid for text placement
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw measurement text
            text = f"{length}m ({width}mm)"
            cv2.putText(output, text, (cx-30, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Print results grouped by category
    print("\nPipe Measurements by Category:")
    categories = sorted(set(pipe['category'] for pipe in results['pipe_measurements']))
    for category in categories:
        print(f"\n{category.upper()} PIPES:")
        category_pipes = [p for p in results['pipe_measurements'] if p['category'] == category]
        for i, pipe in enumerate(category_pipes, 1):
            print(f"Pipe {i}: Width = {pipe['width_mm']}mm, Length = {pipe['length_m']}m")

    # Save the image
    if not os.path.exists('static'):
        os.makedirs('static')
    output_path = 'static/processed_image.jpg'
    cv2.imwrite(output_path, output)

    return output_path