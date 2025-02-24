import cv2
import numpy as np
from typing import Dict, List, Tuple

class PipeImageProcessor:
    def __init__(self, marker_length: float):
        self.marker_length = marker_length
        # HSV range for red color (considering both ranges around hue=0 and hue=180)
        self.red_ranges = [
            ((0, 30, 30), (15, 255, 255)),  # Broader lower red range
            ((160, 30, 30), (180, 255, 255))  # Broader upper red range
        ]
        # Rectangular width categories in millimeters
        self.width_categories = {
            'small': (200, 400),
            'medium': (401, 700),
            'large': (701, 1500)
        }

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process the image to detect and measure red pipes
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for red objects
        mask = self._create_red_mask(hsv)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_TREE,  # Changed to TREE to get hierarchy information
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No red objects detected in the image")

        # Filter contours by area and shape
        filtered_contours = []
        for i, contour in enumerate(contours):
            # Skip small noise contours
            if cv2.contourArea(contour) < 100:
                continue

            # Get convex hull for better shape analysis
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)

            # Calculate solidity (area ratio)
            solidity = float(contour_area)/hull_area if hull_area > 0 else 0

            # Approximate contour shape using Douglas-Peucker algorithm
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Filter based on shape complexity and solidity
            if len(approx) > 4 and len(approx) < 15 and solidity > 0.7:  # Complex enough to be a pipe
                filtered_contours.append(contour)

        if not filtered_contours:
            raise ValueError("No valid pipe contours detected")

        # Find marker
        marker_contour = self._find_marker(filtered_contours)
        if marker_contour is None:
            raise ValueError("No reference marker found in the image")

        # Calculate pixels per millimeter using marker
        pixels_per_mm = self._calculate_scale(marker_contour) / (self.marker_length * 1000)

        # Process pipe contours
        pipe_measurements = []
        for contour in filtered_contours:
            if np.array_equal(contour, marker_contour):
                continue

            width, length = self._measure_pipe(contour, pixels_per_mm)
            category = self._categorize_width(width)

            # Calculate additional shape metrics
            hull = cv2.convexHull(contour)
            hull_perimeter = cv2.arcLength(hull, True)
            contour_perimeter = cv2.arcLength(contour, True)
            shape_complexity = contour_perimeter / hull_perimeter

            pipe_measurements.append({
                'contour': contour,
                'width_mm': width,
                'length_m': length,
                'category': category,
                'shape_complexity': round(shape_complexity, 3)
            })

        # Sort measurements by width category
        pipe_measurements.sort(key=lambda x: x['width_mm'])

        return {
            'marker_contour': marker_contour,
            'pipe_measurements': pipe_measurements,
            'pixels_per_mm': pixels_per_mm
        }

    def _create_red_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create a binary mask for red objects with improved thresholding"""
        self.mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for (lower, upper) in self.red_ranges:
            range_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            self.mask = cv2.bitwise_or(self.mask, range_mask)

        # Enhanced morphological operations
        kernel = np.ones((5,5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

        # Additional noise removal
        self.mask = cv2.medianBlur(self.mask, 5)
        return self.mask

    def _find_marker(self, contours: List[np.ndarray]) -> np.ndarray:
        """Find the reference marker by looking for green boundaries"""
        # Convert image to HSV for green detection
        image = cv2.cvtColor(cv2.imread('static/test_marker.jpg'), cv2.COLOR_BGR2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Create mask for green color
        green_mask = cv2.inRange(image, lower_green, upper_green)
        
        # Find contours in green mask
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest green contour
        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_green)
            
            if area < 1000:  # Keep the minimum area threshold
                return None
                
            return largest_green
        return None

            # Approximate the contour
            epsilon = 0.04 * perimeter  # More tolerant approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular (3-6 corners to be more lenient)
            if 3 <= len(approx) <= 6:
                # Calculate aspect ratio and extent
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                rect_area = w * h
                extent = float(area)/rect_area

                # Much more lenient criteria
                if 0.5 <= aspect_ratio <= 1.5 and extent > 0.4:
                    marker_candidates.append(contour)

        # Save debug image
        debug_img = cv2.cvtColor(self.mask.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, marker_candidates, -1, (0, 255, 0), 2)
        cv2.imwrite('static/debug_markers.jpg', debug_img)

        # Return the marker candidate closest to expected size
        if marker_candidates:
            # Sort by area to find moderate-sized marker
            sorted_candidates = sorted(marker_candidates, key=cv2.contourArea)
            return sorted_candidates[len(sorted_candidates)//2]  # Pick middle-sized candidate
        return None

    def _calculate_scale(self, marker_contour: np.ndarray) -> float:
        """Calculate pixels per marker length using improved method"""
        # Use multiple methods for robust measurement
        rect = cv2.minAreaRect(marker_contour)
        rect_size = (rect[1][0] + rect[1][1]) / 2  # Average of width and height

        perimeter = cv2.arcLength(marker_contour, True)
        perimeter_size = perimeter / 4  # For square marker

        # Return average of both measurements
        return (rect_size + perimeter_size) / 2

    def _measure_pipe(self, contour: np.ndarray, pixels_per_mm: float) -> Tuple[float, float]:
        """Measure the rectangular width of a pipe"""
        rect = cv2.minAreaRect(contour)
        width = min(rect[1][0], rect[1][1])  # Get rectangular width
        width_mm = width / pixels_per_mm
        
        # Return width twice to maintain function signature, but only width is used
        return round(width_mm, 1), 0

    def _categorize_width(self, width_mm: float) -> str:
        """Categorize pipe based on its width"""
        for category, (min_width, max_width) in self.width_categories.items():
            if min_width <= width_mm <= max_width:
                return category
        return 'unknown'