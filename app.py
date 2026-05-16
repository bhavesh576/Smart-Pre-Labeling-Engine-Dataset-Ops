import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import zipfile
import pandas as pd

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
    
    # NEW: Dictionaries to keep track of bad/weird data
    anomalies = {
        'empty': [],
        'overlapping': []
    }
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        
        for index, uploaded_file in enumerate(uploaded_files):
            input_image = Image.open(uploaded_file)
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
                
            results = model(input_image, conf=conf_threshold)
            base_filename = uploaded_file.name.rsplit('.', 1)[0]
            
            img_buffer = io.BytesIO()
            input_image.save(img_buffer, format="JPEG")
            zip_file.writestr(f"{base_filename}.jpg", img_buffer.getvalue())
            
            txt_content = ""
            boxes = results[0].boxes
            
            # 1. Check for Empty Images
            if len(boxes) == 0:
                anomalies['empty'].append(uploaded_file.name)
            
            # 2. Check for Overlapping Boxes (Duplicate Detections)
            has_overlap = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    # Get corner coordinates of the boxes
                    b1 = boxes[i].xyxyn[0] 
                    b2 = boxes[j].xyxyn[0]
                    
                    # Calculate Intersection area
                    inter_x1 = max(b1[0], b2[0])
                    inter_y1 = max(b1[1], b2[1])
                    inter_x2 = min(b1[2], b2[2])
                    inter_y2 = min(b1[3], b2[3])
                    
                    inter_area = max(0, float(inter_x2 - inter_x1)) * max(0, float(inter_y2 - inter_y1))
                    
                    # Calculate areas of individual boxes
                    box1_area = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
                    box2_area = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
                    
                    # Calculate Union area
                    union_area = box1_area + box2_area - inter_area
                    
                    # Calculate IoU (Intersection over Union)
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    # If boxes overlap by more than 85%, flag it!
                    if iou > 0.85:
                        has_overlap = True
                        break # Stop checking this image, it's already flagged
                if has_overlap:
                    break
            
            if has_overlap:
                anomalies['overlapping'].append(uploaded_file.name)

            # Track statistics and generate text file
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id] 
                
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
                    
                total_objects_found += 1
                
                x_center, y_center, width, height = box.xywhn[0]
                txt_content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            zip_file.writestr(f"{base_filename}.txt", txt_content)
            
            annotated_frame = results[0].plot()
            output_image = Image.fromarray(annotated_frame[..., ::-1]) 
            
            with st.expander(f"🖼️ View Result: {uploaded_file.name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(output_image, caption=f"AI Labeled preview", use_container_width=True)
                with col2:
                    st.write("**Extracted Coordinates (.txt format):**")
                    if txt_content:
                        st.code(txt_content, language="text")
                    else:
                        st.write("*No objects detected.*")
            
            current_progress = (index + 1) / len(uploaded_files)
            progress_bar.progress(current_progress)
            progress_text.text(f"Processed {index + 1} of {len(uploaded_files)}")
            
    st.success("✅ Batch processing complete!")
    
    st.markdown("---")
    
    # Put the Dashboard and the QA Alerts side-by-side using columns
    col_dash, col_qa = st.columns([2, 1])
    
    with col_dash:
        st.subheader("📊 Dataset Health Report")
        if total_objects_found > 0:
            st.write(f"**Total Objects Detected:** {total_objects_found}")
            df = pd.DataFrame(list(class_counts.items()), columns=['Object Class', 'Count'])
            df = df.set_index('Object Class')
            st.bar_chart(df)
        else:
            st.warning("No objects were detected in this batch.")
            
    # ---------------------------------------------------------
    # NEW: THE QA ALERTS SECTION
    # ---------------------------------------------------------
    with col_qa:
        st.subheader("⚠️ QA Alerts")
        st.write("Auto-flagged issues for human review:")
        
        if anomalies['empty']:
            st.error(f"**{len(anomalies['empty'])} Empty Images:** \nNo objects found. Might be background noise.")
            with st.expander("View file names"):
                for name in anomalies['empty']:
                    st.write(f"- {name}")
        else:
            st.success("**0 Empty Images.**")
            
        if anomalies['overlapping']:
            st.warning(f"**{len(anomalies['overlapping'])} Overlapping Boxes:** \nAI likely drew duplicate boxes on the same object (IoU > 85%).")
            with st.expander("View file names"):
                for name in anomalies['overlapping']:
                    st.write(f"- {name}")
        else:
            st.success("**0 Overlapping Boxes.**")
    # ---------------------------------------------------------

    st.markdown("---")
    st.download_button(
        label="📦 Download Pro Dataset (.zip with Clean Images & .txt Data)",
        data=zip_buffer.getvalue(),
        file_name="pro_labeled_dataset.zip",
        mime="application/zip",
        use_container_width=True 
    )
