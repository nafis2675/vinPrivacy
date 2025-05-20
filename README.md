# Chassis/VIN Number Detector & Obscurer

A web-based tool that uses PaddleOCR to detect chassis numbers, VINs, and other vehicle identifiers in images and obscure them using blur or mosaic effects. This application was adapted from a Colab notebook to run as a standalone Flask web application.

## Features

- Upload images through an intuitive drag-and-drop interface
- Batch processing for multiple images at once
- Automatic detection of VINs, chassis numbers, and frame numbers
- Manufacturer code detection (WDD, JT, etc.) for more accurate identification
- Options to obscure first, middle, last, or the full identifier
- Special options to obscure only after manufacturer codes or only the codes themselves
- Two obscuring methods: Gaussian blur and mosaic effect
- Adjustable blur intensity and mosaic block size
- Real-time preview of processed images
- Download processed images directly from the web interface
- Detailed detection results
- Debug mode for troubleshooting

## Installation

1. Clone the repository or download the source code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your web browser and go to: http://127.0.0.1:5000/

## Usage

1. Upload one or more images containing vehicle identification information
2. Select which part of the identifier to obscure (First, Middle, Last, Full, After Manufacturer Code, or Only Manufacturer Code)
3. Choose between Blur and Mosaic obscuring methods
4. Adjust the blur intensity or mosaic block size using the sliders
5. Click "Process Image" to detect and obscure identifiers
6. View the results on the right panel
7. Download processed images using the download button

## Detection Strategies

The application uses multiple strategies to detect vehicle identifiers:

1. **VIN Pattern Matching**: Detects standard 17-character VINs
2. **Keyword-based Detection**: Finds text following keywords like "chassis", "VIN", "frame", etc.
3. **Multi-text Block Joining**: Recognizes when an identifier spans multiple text blocks

## Notes

- The application automatically attempts to use GPU if available, but will fall back to CPU processing
- Processed images are saved in the `static/outputs` folder
- Uploaded images are temporarily stored in the `uploads` folder
- For privacy reasons, images are not automatically deleted after processing

## Requirements

- Python 3.7+
- PaddlePaddle
- PaddleOCR
- Flask
- OpenCV
- NumPy
- Pillow
- Matplotlib
