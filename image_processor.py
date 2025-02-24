import cv2
import numpy as np
from typing import Dict, List, Tuple

class PipeImageProcessor:
    def __init__(self, marker_length: float):
        self.marker_length = marker_length
        # HSV range for red color (considering both ranges around hue=0 and hue=180)
        self.red_ranges = [
            ((0, 50, 50), (10, 255, 255)),
            ((170, 50, 50), (180, 255, 255))
        ]
        # Width categories in millimeters
        self.width_categories = {
            'small': (25, 50),
            'medium': (51, 75),
            'large': (76, 100)
        }

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process the image to detect and measure red pipes
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for red objects
        mask = self._create_red_mask(hsv)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find marker (assuming the smallest red object is the reference marker)
        marker_contour = min(contours, key=cv2.contourArea) if contours else None

        if marker_contour is None:
            raise ValueError("No reference marker found in the image")

        # Calculate pixels per millimeter using marker
        pixels_per_mm = self._calculate_scale(marker_contour) / (self.marker_length * 1000)

        # Process pipe contours
        pipe_measurements = []
        for contour in contours:
            if contour is marker_contour:
                continue

            width, length = self._measure_pipe(contour, pixels_per_mm)
            category = self._categorize_width(width)

            pipe_measurements.append({
                'contour': contour,
                'width_mm': width,
                'length_m': length,
                'category': category
            })

        # Sort measurements by width category
        pipe_measurements.sort(key=lambda x: x['width_mm'])

        return {
            'marker_contour': marker_contour,
            'pipe_measurements': pipe_measurements,
            'pixels_per_mm': pixels_per_mm
        }

    def _create_red_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create a binary mask for red objects"""
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for (lower, upper) in self.red_ranges:
            range_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, range_mask)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _calculate_scale(self, marker_contour: np.ndarray) -> float:
        """Calculate pixels per marker length"""
        marker_pixels = cv2.arcLength(marker_contour, True)
        return marker_pixels / 4  # Perimeter divided by 4 for square marker

    def _measure_pipe(self, contour: np.ndarray, pixels_per_mm: float) -> Tuple[float, float]:
        """Measure the width and length of a pipe in millimeters and meters"""
        # Find the minimum area rectangle that bounds the pipe
        rect = cv2.minAreaRect(contour)
        width = min(rect[1][0], rect[1][1])  # Shorter side is the width
        length = max(rect[1][0], rect[1][1])  # Longer side is the length

        # Convert measurements
        width_mm = width / pixels_per_mm
        length_m = (length / pixels_per_mm) / 1000  # Convert mm to m

        return round(width_mm, 1), round(length_m, 2)

    def _categorize_width(self, width_mm: float) -> str:
        """Categorize pipe based on its width"""
        for category, (min_width, max_width) in self.width_categories.items():
            if min_width <= width_mm <= max_width:
                return category
        return 'unknown'