# Vehicle Counting and Classification using YOLOv8 + Norfair

This project uses **YOLOv8** and **Norfair tracking** to detect, classify, and count vehicles in a video feed based on their direction (IN/OUT). The system is optimized for performance and accuracy in real-world traffic video analysis.

## 🚗 Supported Vehicle Classes
* Bicycle
* Motorcycle
* Car
* Bus
* Truck

## 🎯 Key Features
* Real-time vehicle detection and tracking
* Direction-based IN/OUT counting
* Visual overlays with bounding boxes, labels, trails, and velocity vectors
* ROI-based detection zone configuration
* Console logging and summary output for all detected events

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
vehicle-counting-yolov8/
├── vehicleClassification.py       # Main detection and tracking script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Ignores unnecessary files
└── Test Video for Vehicle Counting Model - Indian Road.mp4  # Sample input video
```

## ▶️ Running the Project

1. Ensure your video file is available and correctly named:

```python
VIDEO_PATH = "./Test Video for Vehicle Counting Model - Indian Road.mp4"
```

Modify the path in `vehicleClassification.py` if needed.

2. Run the script:

```bash
python vehicleClassification.py
```

## 🎮 Controls
* Press `q` — Quit
* Press `p` — Pause/Play
* Press `r` — Reset IN/OUT counts

## 📊 Output Summary
* Live window displaying tracking, classification, and direction arrows
* Terminal printout of all vehicle detections
* End-of-video summary report showing counts by class and direction

## 📽️ Demonstration
[Add a video or GIF showcasing the project in action here. You can also link a YouTube video.]

## 🧠 Technologies Used
* YOLOv8 (Ultralytics)
* Norfair
* OpenCV
* NumPy
* Python

## 📄 License
This project is licensed under the **MIT License**.

---

For any questions or suggestions, feel free to open an issue or reach out via GitHub.