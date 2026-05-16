import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import zipfile
import pandas as pd # NEW: We need pandas to format data for the chart

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
    
    # NEW: Create a dictionary to count how many times each class is found
    class_counts = {}
    total_objects_found = 0
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        
        for index, uploaded_file in enumerate(uploaded_files):
            input_image = Image.open(uploaded_file)
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
                
            results = model(input_image, conf=conf_threshold)
            base_filename = uploaded_file.name.rsplit('.', 1)[0]
            
            # Save clean image
            img_buffer = io.BytesIO()
            input_image.save(img_buffer, format="JPEG")
            zip_file.writestr(f"{base_filename}.jpg", img_buffer.getvalue())
            
            txt_content = ""
            boxes = results[0].boxes
            
            # NEW: Track the statistics while we generate the text file
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id] # Get the actual text name (e.g., 'car', 'person')
                
                # Add to our counter
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
                    
                total_objects_found += 1
                
                x_center, y_center, width, height = box.xywhn[0]
                txt_content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            zip_file.writestr(f"{base_filename}.txt", txt_content)
            
            # Show preview
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
    
    # ---------------------------------------------------------
    # NEW: THE DATASET MRI DASHBOARD
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Dataset Health Report")
    
    if total_objects_found > 0:
        st.write(f"**Total Objects Detected:** {total_objects_found}")
        
        # Convert our dictionary into a Pandas DataFrame so Streamlit can graph it
        df = pd.DataFrame(list(class_counts.items()), columns=['Object Class', 'Count'])
        df = df.set_index('Object Class')
        
        # Draw a beautiful bar chart
        st.bar_chart(df)
        
        # Add a quick "Data Engineer" warning if there is heavy imbalance
        st.info("💡 **Dataset Ops Tip:** Look at the chart above. If one bar is massive and the others are tiny, your dataset has a **Class Imbalance**. The AI will become biased towards the tall bar. Consider uploading more images of the rarer objects before training.")
    else:
        st.warning("No objects were detected in this batch. Try lowering the AI Confidence Threshold.")
    # ---------------------------------------------------------

    st.download_button(
        label="📦 Download Pro Dataset (.zip with Clean Images & .txt Data)",
        data=zip_buffer.getvalue(),
        file_name="pro_labeled_dataset.zip",
        mime="application/zip",
        use_container_width=True 
    )
