# 🏙️ Smart City Issue Detection System

Garbage & Pothole Detection using YOLOv8 + Flask

An AI-powered computer vision web application that detects garbage dumps and road potholes from images to support smart city infrastructure monitoring.

---

## Problem
Urban infrastructure issues such as potholes and garbage accumulation negatively impact safety, cleanliness, and quality of life. Traditional manual reporting systems are slow, inconsistent, and reactive. This project automates detection using deep learning–based object detection, enabling:
- Faster identification of infrastructure issues
- Centralized issue tracking
- Data-driven decision-making
- Scalable smart city monitoring solutions

## Features
**🔍 Detection**
- Custom-trained YOLOv8m object detection model
- Detects 🕳️ Potholes and 🗑️ Garbage
- Bounding box visualization
- Confidence-based filtering

**🖥️ Web Application (Flask)**
- Image upload for issue reporting
- Live camera capture support
- Automatic detection summary
- Severity classification (Low / Medium / High)
- Responsive UI using Bootstrap 5

**📊 Analytics & Reports**
- Automatic report saving
- SQLite database integration
- Report history page
- Overall analytics dashboard:
  - Total reports, potholes, and garbage detected
  - No-issue reports
  - Pie chart visualization using Chart.js
- Delete individual reports or all reports

## Architecture
**Application Workflow:**
1. User uploads an image or captures via camera
2. YOLOv8 model performs object detection
3. Bounding boxes are drawn on the image
4. Issues are counted per class
5. Severity is calculated automatically
6. Report is saved to database
7. Analytics dashboard updates in real-time

**Project Structure:**
```
AI-Smart-City/
│
├── app.py                     # Flask application (production inference)
├── yolov8m.pt                 # Custom-trained YOLOv8 model
├── reports.db                 # SQLite database (report history)
│
├── templates/
│   ├── index.html             # Main detection UI
│   └── history.html           # Report history & analytics
│
├── static/
│   ├── style.css              # UI styling
│   ├── script.js              # Frontend logic (upload, camera, charts)
│   └── reports/               # Saved report images (ignored in Git)
│
├── model-training.ipynb       # Training notebook (Google Colab)
├── notebook.ipynb             # Evaluation & inference testing
│
├── requirements.txt           # Dependencies
├── .gitignore                 # Ignored runtime & generated files
└── README.md                  # Project documentation
```

## Tech Stack
| Category | Technology |
| --- | --- |
| Language | Python 3.9+ |
| Model | YOLOv8 (Ultralytics) |
| Backend | Flask |
| Frontend | HTML, CSS, Bootstrap 5 |
| Charts | Chart.js |
| Database | SQLite |
| Dataset | Roboflow |
| Image Processing | Pillow |
| Training | Google Colab (GPU) |

## Dataset
**Platform:** Roboflow

**Classes:** `pothole`, `garbage`

**Annotation & preprocessing:**
- Bounding box annotation
- Data augmentation (flip, rotate, brightness, blur)
- Train / validation / test split
- Dataset size: ~2,000+ images

## Model
**Architecture:** YOLOv8m
**Framework:** Ultralytics YOLO
**Environment:** Google Colab (GPU)
**Training Notebook:** `model-training.ipynb`

⚠️ *Note: Training notebooks are GPU-oriented and not intended for local execution without GPU support.*

## Results
**Model Evaluation (Current):**
| Metric | Value |
| --- | --- |
| Precision | ~0.62 |
| Recall | ~0.58 |
| mAP@0.5 | ~0.60 |

*Metrics are expected to improve with dataset expansion and tuning.*

**Example Output:**
- Detected: 2 potholes and 1 garbage
- Severity: Medium
- Visual bounding boxes on image
- Summary: “Total of 3 issues detected: 2 potholes and 1 garbage.”

## API
*(Not yet implemented - planned for Future Enhancements)*
- Future REST API for mobile app integration.

**👷 Municipal Worker Panel**
- Dedicated field worker dashboard & task management
- Worker login & session-backed authorization
- Problems faced, tools used, and team members tracking
- After-repair photo upload & browser camera capture
- Interactive BEFORE | AFTER visual evidence comparison

## 🌐 Application Working URLs
- **Citizen Detection Command Center**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Report History & Heatmap**: [http://127.0.0.1:8000/history](http://127.0.0.1:8000/history)
- **AI Performance Analytics**: [http://127.0.0.1:8000/performance](http://127.0.0.1:8000/performance)
- **Worker Portal Login**: [http://127.0.0.1:8000/worker/login](http://127.0.0.1:8000/worker/login)
- **Worker Dashboard**: [http://127.0.0.1:8000/worker/dashboard](http://127.0.0.1:8000/worker/dashboard)
- **Worker Profile**: [http://127.0.0.1:8000/worker/profile](http://127.0.0.1:8000/worker/profile)
- **Worker Completed Work**: [http://127.0.0.1:8000/worker/history](http://127.0.0.1:8000/worker/history)

### 🔑 Test Worker Accounts
| Worker ID | Name | Department | Password |
| :--- | :--- | :--- | :--- |
| **`WRK-1024`** | Rahul Sharma | Roads & Infrastructure | `password123` |
| **`WRK-1001`** | Suresh Kumar | Department of Environment | `password123` |
| **`WRK-1005`** | Anita Patel | Sanitation | `password123` |
| **`WRK-1010`** | Vikram Singh | General Municipal Services | `password123` |

## Installation
**1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt psutil
```

## Usage
**2️⃣ Run the Application**
```bash
python app.py
```
**3️⃣ Open in Browser**
http://127.0.0.1:8000

## Future Improvements
**🚧 Future Enhancements**
- Real-time video & CCTV stream detection
- GPS-based issue mapping
- Role-based authentication (admin / user)
- REST API for mobile app integration
- Cloud deployment (AWS / GCP)

**📈 Accuracy Improvement Plan**
- Expand dataset to 5,000+ images
- Add hard-negative samples
- YOLOv8 hyperparameter tuning
- Train with early stopping
- Experiment with YOLOv8l architecture

**Challenges & Limitations**
- Image-based inference only; no real-time video stream yet
- CPU inference latency compared to GPU
- High variability in pothole shapes and lighting
- Class imbalance between pothole and garbage

---
**Author:** Shreeyash Paraj
Data Science Intern | AI & Backend Development
*Project built to demonstrate real-world ML system design & deployment*