import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import zipfile

# UI UPGRADE: Make the layout wide
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
st.write("Upload a batch of images. The AI will detect objects and export them in **YOLO text format** (ready for Human-in-the-Loop correction).")
st.markdown("---")

uploaded_files = st.file_uploader("Drag and drop your images here...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"### Processing {len(uploaded_files)} images...")
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        
        for index, uploaded_file in enumerate(uploaded_files):
            
            # 1. Open the image and ensure it's in standard RGB format
            input_image = Image.open(uploaded_file)
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
                
            # Run the AI Model
            results = model(input_image, conf=conf_threshold)
            
            # Remove the original file extension (like .jpg) to use the base name
            base_filename = uploaded_file.name.rsplit('.', 1)[0]
            
            # --- NEW: PRO DATA EXPORT LOGIC ---
            
            # A) Save the CLEAN, original image to the zip file (Labelers need the raw image)
            img_buffer = io.BytesIO()
            input_image.save(img_buffer, format="JPEG")
            zip_file.writestr(f"{base_filename}.jpg", img_buffer.getvalue())
            
            # B) Extract exact AI coordinates and create a YOLO-formatted .txt file
            txt_content = ""
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0])
                # xywhn gives us standard normalized coordinates: Center X, Center Y, Width, Height
                x_center, y_center, width, height = box.xywhn[0]
                txt_content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            # C) Save the text file inside the zip right next to the image
            zip_file.writestr(f"{base_filename}.txt", txt_content)
            
            # ----------------------------------
            
            # Still generate the painted image just to show on the website UI
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
            
            # Update Progress Bar
            current_progress = (index + 1) / len(uploaded_files)
            progress_bar.progress(current_progress)
            progress_text.text(f"Processed {index + 1} of {len(uploaded_files)}")
            
    st.success("✅ Batch processing complete! Your dataset is ready.")
    
    st.download_button(
        label="📦 Download Pro Dataset (.zip with Clean Images & .txt Data)",
        data=zip_buffer.getvalue(),
        file_name="pro_labeled_dataset.zip",
        mime="application/zip",
        use_container_width=True 
    )
