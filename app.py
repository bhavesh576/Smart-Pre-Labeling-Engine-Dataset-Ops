import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import zipfile
import pandas as pd
import cv2
import numpy as np

st.set_page_config(page_title="Smart Pre-Labeling Engine", page_icon="🎯", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

with st.sidebar:
    st.header("⚙️ AI Settings")
    conf_threshold = st.slider("AI Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    st.info("💡 **Tip:** \n* **High Confidence (0.8+):** Fewer boxes, but highly accurate.\n* **Low Confidence (0.2):** Catches everything, but might make mistakes.")

st.title("🎯 Smart Pre-Labeling Engine (Pro Export)")
st.write("Upload a batch of images to auto-label and analyze your dataset.")
st.markdown("---")

uploaded_files = st.file_uploader("Drag and drop your images here...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"### Processing {len(uploaded_files)} images...")
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    zip_buffer = io.BytesIO()
    
    class_counts = {}
    total_objects_found = 0
    
    anomalies = {
        'empty': [],
        'overlapping': []
    }
    
    scenarios = {
        'low_light': 0,
        'blurry': 0,
        'crowded': 0,
        'small_objects_only': 0
    }
    
    ui_preview_data = []
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        
        for index, uploaded_file in enumerate(uploaded_files):
            input_image = Image.open(uploaded_file)
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
                
            img_cv = np.array(input_image)
            
            # ⚡ EXTREME SPEED FIX 1: Shrink to 320x320 for OpenCV math
            img_cv_fast = cv2.resize(img_cv, (320, 320))
            
            gray = cv2.cvtColor(img_cv_fast, cv2.COLOR_RGB2GRAY)
            
            brightness = np.mean(gray)
            if brightness < 80:
                scenarios['low_light'] += 1
                
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:  
                scenarios['blurry'] += 1
                
            # ⚡ EXTREME SPEED FIX 2: Force YOLO to run at a smaller size (imgsz=320)
            results = model(input_image, conf=conf_threshold, imgsz=320)
            base_filename = uploaded_file.name.rsplit('.', 1)[0]
            
            img_buffer = io.BytesIO()
            input_image.save(img_buffer, format="JPEG")
            zip_file.writestr(f"{base_filename}.jpg", img_buffer.getvalue())
            
            txt_content = ""
            boxes = results[0].boxes
            
            if len(boxes) > 15:
                scenarios['crowded'] += 1
            
            if len(boxes) == 0:
                anomalies['empty'].append(uploaded_file.name)
            
            has_overlap = False
            max_box_area_ratio = 0.0 
            
            for i in range(len(boxes)):
                b_current = boxes[i].xywhn[0]
                box_area_ratio = float(b_current[2] * b_current[3])
                if box_area_ratio > max_box_area_ratio:
                    max_box_area_ratio = box_area_ratio
                
                for j in range(i + 1, len(boxes)):
                    b1 = boxes[i].xyxyn[0] 
                    b2 = boxes[j].xyxyn[0]
                    
                    inter_x1 = max(b1[0], b2[0])
                    inter_y1 = max(b1[1], b2[1])
                    inter_x2 = min(b1[2], b2[2])
                    inter_y2 = min(b1[3], b2[3])
                    
                    inter_area = max(0, float(inter_x2 - inter_x1)) * max(0, float(inter_y2 - inter_y1))
                    box1_area = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
                    box2_area = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > 0.85:
                        has_overlap = True
                        break 
                if has_overlap:
                    break
            
            if has_overlap:
                anomalies['overlapping'].append(uploaded_file.name)
                
            if len(boxes) > 0 and max_box_area_ratio < 0.05:
                scenarios['small_objects_only'] += 1

            total_conf = 0
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id] 
                total_conf += float(box.conf[0])
                
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
                    
                total_objects_found += 1
                
                x_center, y_center, width, height = box.xywhn[0]
                txt_content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            avg_conf = (total_conf / len(boxes)) if len(boxes) > 0 else 0.0
            
            zip_file.writestr(f"{base_filename}.txt", txt_content)
            
            annotated_frame = results[0].plot()
            output_image = Image.fromarray(annotated_frame[..., ::-1]) 
            
            ui_preview_data.append({
                'filename': uploaded_file.name,
                'image': output_image,
                'txt_content': txt_content,
                'avg_conf': avg_conf,
                'box_count': len(boxes)
            })
            
            current_progress = (index + 1) / len(uploaded_files)
            progress_bar.progress(current_progress)
            progress_text.text(f"Processed {index + 1} of {len(uploaded_files)}")
            
    st.success("✅ Batch processing complete!")
    st.markdown("---")
    
    st.subheader("🕵️ Dataset Blind Spot Finder")
    st.write("Auditing your dataset for real-world scenarios to prevent model failure.")
    
    total_imgs = len(uploaded_files)
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric(label="Night / Low Light", value=f"{scenarios['low_light']} imgs", delta=f"{round((scenarios['low_light']/total_imgs)*100, 1)}%", delta_color="off")
    with col_s2:
        st.metric(label="Blurry / Out of Focus", value=f"{scenarios['blurry']} imgs", delta=f"{round((scenarios['blurry']/total_imgs)*100, 1)}%", delta_color="off")
    with col_s3:
        st.metric(label="Crowded (>15 objects)", value=f"{scenarios['crowded']} imgs", delta=f"{round((scenarios['crowded']/total_imgs)*100, 1)}%", delta_color="off")
    with col_s4:
        st.metric(label="Distant / Small Objects", value=f"{scenarios['small_objects_only']} imgs", delta=f"{round((scenarios['small_objects_only']/total_imgs)*100, 1)}%", delta_color="off")
    
    st.markdown("---")
    
    # --- NEW: FAILURE PREDICTION REPORT ---
    st.subheader("🚨 'Why Will My Model Fail?' - Predictive Risk Report")
    st.write("Automated risk assessment based on dataset representation gaps. Models require at least 15% coverage of an edge case to generalize safely.")
    
    # We want at least 15% of the dataset to represent a difficult scenario for robustness
    target_coverage_pct = 15.0 
    risks_found = False
    
    def calculate_and_display_risk(scenario_name, count, total_images):
        pct = (count / total_images) * 100 if total_images > 0 else 0
        
        if pct < target_coverage_pct:
            # Calculate a harsh failure probability based on how far they are from the 15% target
            fail_prob = min(99, int(100 - (pct / target_coverage_pct * 100)))
            missing_imgs = max(10, int((target_coverage_pct / 100 * total_images) - count))
            
            st.error(f"**⚠️ {fail_prob}% Failure Probability on: {scenario_name}** \n\n"
                     f"**Analysis:** Only {count} images ({pct:.1f}%) feature this condition. The model will likely collapse or hallucinate when encountering this in production.\n\n"
                     f"**Action Required:** Source a minimum of **{missing_imgs} additional images** featuring {scenario_name.lower()} before initiating model training.")
            return True
        return False
        
    risks_found |= calculate_and_display_risk("Low-Light / Nighttime Environments", scenarios['low_light'], total_imgs)
    risks_found |= calculate_and_display_risk("Blurry / High-Motion Environments", scenarios['blurry'], total_imgs)
    risks_found |= calculate_and_display_risk("Highly Cluttered / Crowded Scenes", scenarios['crowded'], total_imgs)
    risks_found |= calculate_and_display_risk("Distant / Extremely Small Objects", scenarios['small_objects_only'], total_imgs)
    
    if not risks_found:
        st.success("✅ **Dataset Risk Assessment Passed!** Your dataset has excellent coverage across diverse real-world edge cases. Ready for training.")
    # ------------------------------------------------
    
    st.markdown("---")
    
    col_dash, col_qa = st.columns([2, 1])
    
    with col_dash:
        st.subheader("📊 Class Distribution")
        if total_objects_found > 0:
            df = pd.DataFrame(list(class_counts.items()), columns=['Object Class', 'Count'])
            df = df.set_index('Object Class')
            st.bar_chart(df)
        else:
            st.warning("No objects were detected in this batch.")
            
    with col_qa:
        st.subheader("⚠️ QA Alerts")
        if anomalies['empty']:
            st.error(f"**{len(anomalies['empty'])} Empty Images**")
        else:
            st.success("**0 Empty Images.**")
            
        if anomalies['overlapping']:
            st.warning(f"**{len(anomalies['overlapping'])} Overlapping Boxes** (IoU > 85%)")
        else:
            st.success("**0 Overlapping Boxes.**")

    st.markdown("---")
    
    st.subheader("🧠 Smart Review Queue (Active Learning)")
    st.write("Images sorted by **Lowest Average Confidence**.")
    
    ui_preview_data.sort(key=lambda x: x['avg_conf'] if x['box_count'] > 0 else 2.0)
    
    for data in ui_preview_data:
        conf_display = f"{data['avg_conf']:.2f}" if data['box_count'] > 0 else "N/A (Empty)"
        with st.expander(f"🖼️ {data['filename']} | Avg Confidence: {conf_display} | Objects: {data['box_count']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(data['image'], caption="AI Labeled preview", use_container_width=True)
            with col2:
                if data['txt_content']:
                    st.code(data['txt_content'], language="text")
                else:
                    st.write("*No objects detected.*")

    st.markdown("---")
    st.download_button(
        label="📦 Download Pro Dataset (.zip with Clean Images & .txt Data)",
        data=zip_buffer.getvalue(),
        file_name="pro_labeled_dataset.zip",
        mime="application/zip",
        use_container_width=True 
    )
