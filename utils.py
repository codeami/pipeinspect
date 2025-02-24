import cv2
import numpy as np
from typing import Dict
import os

def display_results(image: np.ndarray, results: Dict) -> str:
    """Process the image with measurements and save results"""
    # Create a copy of the image for drawing
    output = image.copy()

    # Draw marker contour with label
    marker_contour = results['marker_contour']
    cv2.drawContours(output, [marker_contour], -1, (0, 255, 0), 2)
    # Add marker label
    x, y, w, h = cv2.boundingRect(marker_contour)
    cv2.putText(output, "Reference Marker", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

            # Draw width line (blue)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            width_p1 = tuple(box[0])
            width_p2 = tuple(box[1])
            cv2.line(output, width_p1, width_p2, (255, 0, 0), 3)  # Blue line

            # Draw length line (pink)
            length_p1 = tuple(box[1])
            length_p2 = tuple(box[2])
            cv2.line(output, length_p1, length_p2, (255, 0, 255), 3)  # Pink line

            # Draw arrows
            # White arrow for width
            arrow_start = ((width_p1[0] + width_p2[0])//2, (width_p1[1] + width_p2[1])//2)
            cv2.arrowedLine(output, arrow_start, 
                          (arrow_start[0], arrow_start[1] - 50), 
                          (255, 255, 255), 2, tipLength=0.5)  # White arrow

            # Yellow arrow for length
            arrow_start = ((length_p1[0] + length_p2[0])//2, (length_p1[1] + length_p2[1])//2)
            cv2.arrowedLine(output, arrow_start,
                          (arrow_start[0] + 50, arrow_start[1]), 
                          (0, 255, 255), 2, tipLength=0.5)  # Yellow arrow

            # Position measurements in columns
            if i % 2 == 0:  # Left column
                text_x = 50
            else:  # Right column
                text_x = output.shape[1] - 400
            
            text_y = 100 + (i // 2) * 100  # New row every two measurements
            text = f"{length}m ({width}mm)"
            cv2.putText(output, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 4)  # Bright green

    # Add legend
    legend_y = max(text_y + 100, 100)
    cv2.putText(output, "Pipe Categories:", (30, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 4)
    for i, (category, color) in enumerate(category_colors.items(), 1):
        cv2.putText(output, f"- {category.title()}", (60, legend_y + i*100),
                   cv2.FONT_HERSHEY_SIMPLEX, 5.0, color, 4)

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