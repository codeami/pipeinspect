#!/usr/bin/env python3
import argparse
import cv2
import sys
from image_processor import PipeImageProcessor
from utils import display_results

def main():
    parser = argparse.ArgumentParser(description='Measure red pipes in construction site images')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--marker-length', type=float, default=1.0,
                        help='Known length of the marker in meters (default: 1.0)')
    args = parser.parse_args()

    try:
        # Read image
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError("Could not read the image")

        # Process image
        processor = PipeImageProcessor(marker_length=args.marker_length)
        results = processor.process_image(image)

        # Display results
        display_results(image, results)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
