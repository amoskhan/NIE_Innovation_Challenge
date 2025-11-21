# Mediapipe Webcam Pose Detection

This project utilizes MediaPipe's Pose Landmarker to detect and visualize human poses in real-time using a webcam. The application captures video from the webcam, processes it to identify key landmarks on the human body, and displays the results with visual annotations.

## Project Structure

```
mediapipe-webcam-pose
├── src
│   ├── mediapipevideo.py       # Main logic for video capture and pose detection
│   └── utils.py                # Utility functions for data processing
├── models
│   └── pose_landmarker_full.task # Model file for pose detection
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files and directories to ignore in Git
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mediapipe-webcam-pose
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model file:**
   Ensure that the `pose_landmarker_full.task` file is located in the `models` directory.

## Usage

To run the application, execute the following command:

```bash
python src/mediapipevideo.py
```

You can specify the webcam index and running mode using command-line arguments. For example:

```bash
python src/mediapipevideo.py --cam 0 --mode VIDEO
```

## Key Features

- Real-time pose detection using webcam input.
- Visualization of detected landmarks on the video feed.
- Calculation of the hip centroid for movement tracking.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.