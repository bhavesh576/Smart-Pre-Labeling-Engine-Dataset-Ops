# 🎯 Smart Pre-Labeling Engine & Dataset Ops Platform

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://yolowebgit-ggppnfjm9pavtjflwbvu4m.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end, batch-processing dataset engineering platform built to solve the manual data annotation bottleneck in computer vision workflows. 

Unlike standard object detection wrappers, this application focuses on **Data-Centric AI** principles. It doesn't just run inference; it actively audits dataset health, detects labeling anomalies using IoU mathematics, uncovers environmental blind spots, and optimizes human-in-the-loop (HITL) review queues using active learning concepts.

## ✨ Core Features

* **⚡ Batch Inference & Pro Export:** Upload hundreds of images at once. The engine runs YOLOv8 inference and securely packages the original images alongside `YOLO-formatted .txt` coordinate files into a single `.zip` download, ready for professional annotation software (like Label Studio).
* **📊 Dataset MRI (Health Analytics):** Automatically generates a statistical dashboard of the processed batch, instantly highlighting class imbalances so engineers can source better data before training.
* **🕵️ Dataset Blind Spot Finder:** Uses OpenCV to calculate pixel-level statistics (e.g., Variance of Laplacian for blurriness, mean pixel intensity for lighting). It actively audits the dataset for missing real-world scenarios (Night/Low Light, Blurry, Crowded, Distant Objects) to prevent model failure in production.
* **⚠️ Automated QA Gatekeeper:** Programmatically protects dataset integrity by flagging edge cases:
  * **IoU Overlap Detection:** Flags images where the AI likely stuttered and drew duplicate bounding boxes (Intersection over Union > 85%).
  * **Background Mining:** Flags frames with zero detections to ensure negative background samples are handled correctly.
* **🧠 Active Learning "Smart Review" Queue:** Decouples inference from UI rendering. Images are mathematically sorted by their **Lowest Average Confidence Score**, forcing human labelers to review the hardest, most ambiguous images first rather than wasting time on easy 99%-confidence detections.
* **🚀 Cloud-Optimized Processing:** Implements dynamic image resizing (320x320) before OpenCV mathematical analysis, allowing heavy computer vision pixel math to run at lightning speed on free-tier, CPU-only cloud environments.

## 📸 Dashboard Preview

*(Add your screenshots here! Take screenshots of your Dataset Health Bar Chart, the Blind Spot Finder, your QA Alerts box, and the Smart Review Queue from your live app and drag them into this section.)*

## 🛠️ Tech Stack
* **Frontend/UI:** [Streamlit](https://streamlit.io/) (Deployed via Streamlit Community Cloud)
* **Deep Learning Inference:** [Ultralytics YOLOv8](https://docs.ultralytics.com/)
* **Image Mathematics:** OpenCV (`cv2`)
* **Data Manipulation:** Pandas, NumPy, PIL (Python Imaging Library)

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
git clone [https://github.com/bhavesh576/yoloweb.git](https://github.com/bhavesh576/yoloweb.git)
cd yoloweb
```

2. **Install the required dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch the Streamlit App**
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

## 💡 Why I Built This
In real-world ML pipelines, 80% of the work is dataset curation, not model tuning. I built this tool to automate the repetitive "first draft" of bounding box creation while simultaneously giving MLOps teams visibility into data drift, class imbalance, missing real-world scenarios, and model hallucinations. 

https://yolowebgit-ggppnfjm9pavtjflwbvu4m.streamlit.app/
---
*Developed by Bhavesh | Built for modern computer vision workflows.*
