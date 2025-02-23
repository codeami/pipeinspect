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

        # Calculate pixels per meter using marker
        pixels_per_meter = self._calculate_scale(marker_contour)
        
        # Process pipe contours
        pipe_measurements = []
        for contour in contours:
            if contour is marker_contour:
                continue
                
            length = self._measure_pipe(contour, pixels_per_meter)
            pipe_measurements.append({
                'contour': contour,
                'length': length
            })

        return {
            'marker_contour': marker_contour,
            'pipe_measurements': pipe_measurements,
            'pixels_per_meter': pixels_per_meter
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
        """Calculate pixels per meter using the reference marker"""
        marker_pixels = cv2.arcLength(marker_contour, True)
        return marker_pixels / self.marker_length

    def _measure_pipe(self, contour: np.ndarray, pixels_per_meter: float) -> float:
        """Measure the length of a pipe in meters"""
        # Find the minimum area rectangle that bounds the pipe
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get the length of the pipe (longest side of the rectangle)
        width = rect[1][0]
        height = rect[1][1]
        length_pixels = max(width, height)
        
        # Convert to meters
        length_meters = length_pixels / pixels_per_meter
        return round(length_meters, 2)
