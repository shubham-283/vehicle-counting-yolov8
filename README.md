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

### Requirements.txt

```txt
ultralytics
norfair
opencv-python
numpy
torch
torchvision
```

## 📁 Project Structure

```
vehicle-counting-yolov8/
├── vehicleClassification.py       # Main detection and tracking script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── LICENSE                        # MIT License file
├── .gitignore                     # Ignores unnecessary files
├── OutputScreenshots/             # Sample output screenshots
│   ├── output1.png               # Multi-vehicle detection example
│   ├── output2.png               # Mixed vehicle types detection
│   ├── output3.png               # Multi-class detection example
│   └── output4.png               # Advanced tracking demonstration
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

## 📊 Sample Output Results

Here are some examples of the real-time detection, tracking, and counting in action.

---

<p align="center">
  <img src="OutputScreenshots/output%201.png" width="45%" />
  <img src="OutputScreenshots/output%202.png" width="45%" />
</p>
<p align="center"><b>Image 1:</b> Initial detection | <b>Image 2:</b> Tracking vehicles leaving the frame</p>

---

<p align="center">
  <img src="OutputScreenshots/output%203.png" width="45%" />
  <img src="OutputScreenshots/output%204.png" width="45%" />
</p>
<p align="center"><b>Image 3:</b> Continuous multi-class tracking | <b>Image 4:</b> Handling occlusion and dense traffic</p>

---

<p align="center">
  <img src="OutputScreenshots/output%205.png" width="45%" />
  <img src="OutputScreenshots/output%206.png" width="45%" />
</p>
<p align="center"><b>Image 5:</b> Long-term tracking accuracy | <b>Image 6:</b> Final summary report</p>

---


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
