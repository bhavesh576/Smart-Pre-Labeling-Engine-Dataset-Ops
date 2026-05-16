import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import zipfile

# 1. UI UPGRADE: Make the layout wide and add a custom browser tab icon
st.set_page_config(page_title="Smart Pre-Labeling Engine", page_icon="🎯", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. UI UPGRADE: Create a Sidebar Control Panel
with st.sidebar:
    st.header("⚙️ AI Settings")
    st.write("Adjust how strict the AI should be when finding objects.")
    
    # This slider directly controls the YOLO model's confidence threshold
    conf_threshold = st.slider("AI Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    
    st.info("💡 **Tip:** \n* **High Confidence (0.8+):** Fewer boxes, but highly accurate.\n* **Low Confidence (0.2):** Catches everything, but might make mistakes.")

# Main Screen Layout
st.title("🎯 Smart Pre-Labeling Engine")
st.write("Upload a batch of images to automatically detect and label objects.")
st.markdown("---")

uploaded_files = st.file_uploader("Drag and drop your images here...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"### Processing {len(uploaded_files)} images...")
    
    # 3. UI UPGRADE: Live Progress Bar
    progress_text = st.empty() # Placeholder for text
    progress_bar = st.progress(0) # Start the bar at 0%
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        
        # We use 'enumerate' to count which image we are currently on
        for index, uploaded_file in enumerate(uploaded_files):
            
            input_image = Image.open(uploaded_file)
            
            # We tell the AI to use the confidence number from the sidebar slider
            results = model(input_image, conf=conf_threshold)
            
            annotated_frame = results[0].plot()
            output_image = Image.fromarray(annotated_frame[..., ::-1]) 
            
            img_buffer = io.BytesIO()
            output_image.save(img_buffer, format="JPEG")
            zip_file.writestr(f"labeled_{uploaded_file.name}", img_buffer.getvalue())
            
            # 4. UI UPGRADE: Hide results in a clean, click-to-open expander
            with st.expander(f"🖼️ View Result: {uploaded_file.name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(input_image, caption="Original", use_container_width=True)
                with col2:
                    st.image(output_image, caption=f"AI Labeled (Threshold: {conf_threshold})", use_container_width=True)
            
            # Update the progress bar math
            current_progress = (index + 1) / len(uploaded_files)
            progress_bar.progress(current_progress)
            progress_text.text(f"Processed {index + 1} of {len(uploaded_files)}")
            
    st.success("✅ Batch processing complete! All images are labeled and ready.")
    
    # Make the download button stretch across the whole screen so it's obvious
    st.download_button(
        label="📦 Download All Labeled Images (.zip)",
        data=zip_buffer.getvalue(),
        file_name="labeled_dataset.zip",
        mime="application/zip",
        use_container_width=True 
    )
