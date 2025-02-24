#!/usr/bin/env python3
import argparse
import cv2
import sys
import os
from image_processor import PipeImageProcessor
from utils import display_results

def main():
    parser = argparse.ArgumentParser(description='Measure red pipes in construction site images')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--marker-length', type=float, default=1.0,
                        help='Known length of the marker in meters (default: 1.0)')
    args = parser.parse_args()

    try:
        # Validate image path
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image file not found: {args.image_path}")

        if not os.path.isfile(args.image_path):
            raise ValueError(f"Path is not a file: {args.image_path}")

        # Read image
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Could not read image: {args.image_path}. Make sure it's a valid image file.")

        # Process image
        processor = PipeImageProcessor(marker_length=args.marker_length)
        results = processor.process_image(image)

        # Display results
        output_path = display_results(image, results)
        print(f"\nProcessed image saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()