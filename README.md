# 🎯 Smart Pre-Labeling Engine & Dataset Ops Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end, batch-processing dataset engineering platform built to solve the manual data annotation bottleneck in computer vision workflows. 

Unlike standard object detection wrappers, this application focuses on **Data-Centric AI** principles. It doesn't just run inference; it actively audits dataset health, detects labeling anomalies using IoU mathematics, and optimizes human-in-the-loop (HITL) review queues using active learning concepts.

## ✨ Core Features

* **⚡ Batch Inference & Pro Export:** Upload hundreds of images at once. The engine runs YOLOv8 inference and securely packages the original images alongside `YOLO-formatted .txt` coordinate files into a single `.zip` download, ready for professional annotation software (like Label Studio).
* **📊 Dataset MRI (Health Analytics):** Automatically generates a statistical dashboard of the processed batch, instantly highlighting class imbalances so engineers can source better data before training.
* **⚠️ Automated QA Gatekeeper:** Programmatically protects dataset integrity by flagging edge cases:
  * **IoU Overlap Detection:** Flags images where the AI likely stuttered and drew duplicate bounding boxes (Intersection over Union > 85%).
  * **Background Mining:** Flags frames with zero detections to ensure negative background samples are handled correctly.
* **🧠 Active Learning "Smart Review" Queue:** Decouples inference from UI rendering. Images are mathematically sorted by their **Lowest Average Confidence Score**, forcing human labelers to review the hardest, most ambiguous images first rather than wasting time on easy 99%-confidence detections.

## 📸 Dashboard Preview

*(Add your screenshots here! Take screenshots of your Dataset Health Bar Chart, your QA Alerts box, and the Smart Review Queue from your live app and drag them into this section.)*

## 🛠️ Tech Stack
* **Frontend/UI:** [Streamlit](https://streamlit.io/) (Deployed via Streamlit Community Cloud)
* **Computer Vision:** [Ultralytics YOLOv8](https://docs.ultralytics.com/)
* **Data Manipulation & Math:** Pandas, NumPy, PIL (Python Imaging Library)

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/ai-data-labeling-engine.git](https://github.com/YOUR_GITHUB_USERNAME/ai-data-labeling-engine.git)
cd ai-data-labeling-engine
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
In real-world ML pipelines, 80% of the work is dataset curation, not model tuning. I built this tool to automate the repetitive "first draft" of bounding box creation while simultaneously giving MLOps teams visibility into data drift, class imbalance, and model hallucinations. 

---
*Developed by Bhavesh | Built for modern computer vision workflows.*
