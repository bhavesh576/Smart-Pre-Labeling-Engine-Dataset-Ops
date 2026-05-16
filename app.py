import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import zipfile

# 1. Load the pre-trained AI model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. Set up the Website Interface
st.title("Smart Pre-Labeling Engine 🚀")
st.write("Upload a batch of images, and our AI will automatically detect and label objects.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} images...")
    
    # NEW: Create a temporary "folder" in the computer's memory to hold our zip file
    zip_buffer = io.BytesIO()
    
    # NEW: Open the zip file so we can start putting images inside it
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        
        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.write(f"**File:** {uploaded_file.name}")
            
            input_image = Image.open(uploaded_file)
            results = model(input_image)
            
            annotated_frame = results[0].plot()
            output_image = Image.fromarray(annotated_frame[..., ::-1]) 
            
            # NEW: Save the finished image into the zip file we opened earlier
            img_buffer = io.BytesIO()
            output_image.save(img_buffer, format="JPEG")
            zip_file.writestr(f"labeled_{uploaded_file.name}", img_buffer.getvalue())
            
            # Show the images on the screen
            col1, col2 = st.columns(2)
            with col1:
                st.image(input_image, caption="Original", use_container_width=True)
            with col2:
                st.image(output_image, caption="AI Labeled", use_container_width=True)
                
    st.success("Batch processing complete! All images are labeled.")
    
    # NEW: Create the actual Download Button on the website
    st.download_button(
        label="📦 Download All Labeled Images (.zip)",
        data=zip_buffer.getvalue(),
        file_name="labeled_dataset.zip",
        mime="application/zip"
    )