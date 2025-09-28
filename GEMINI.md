# Project Overview

This project is a Python-based OCR (Optical Character Recognition) application. It captures video from a connected device, detects when the screen is stable, and then performs OCR on the stable image. The application has a graphical user interface (GUI) built with PyQt6, allowing users to view the live video feed, see the OCR results, and configure various settings.

The project has evolved through several phases, starting with basic video capture and progressively adding more advanced features such as:

*   Motion detection to trigger OCR only when the screen is stable.
*   Multithreading to perform OCR in a separate thread, ensuring a smooth user interface.
*   Image preprocessing to improve OCR accuracy.
*   Integration with the EasyOCR library for high-quality text recognition.
*   A GUI to provide a user-friendly experience.
*   The ability to process PDF files.

## Key Technologies

*   **Python**: The core programming language.
*   **OpenCV**: For video capture, image processing, and motion detection.
*   **EasyOCR**: The primary OCR engine.
*   **PyQt6**: For the graphical user interface.
*   **PyMuPDF**: For PDF processing.

# Building and Running

This project does not have a formal build process. It can be run directly from the command line.

## Dependencies

The project requires the following Python libraries:

*   `opencv-python`
*   `pyqt6`
*   `easyocr`
*   `numpy`
*   `Pillow`
*   `PyMuPDF`

You can install these dependencies using pip:

```bash
pip install opencv-python pyqt6 easyocr numpy Pillow PyMuPDF
```

## Running the Application

To run the main application, execute the `app.py` file:

```bash
python app.py
```

This will open the main application window, where you can select the desired mode and start the OCR process.

## Development Conventions

*   The project is structured into several Python files, with `app.py` being the main entry point.
*   The code is well-commented, with clear explanations of the different parts of the application.
*   The project uses a `config.json` file for configuration, allowing users to easily modify settings without changing the code.
*   The project includes a `corrections.txt` file to fix common OCR errors.
