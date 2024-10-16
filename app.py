from transformers import pipeline
import streamlit as st
from PIL import Image

st.title("Image Background Removal App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  
  # Put uploaded and processed images side-by-side
  col1, col2 = st.columns(2)
  with col1:
    st.image(image, caption='Uploaded Image.', use_column_width=True)
  with col2:
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    pillow_mask = pipe(image, return_mask = True) # outputs a pillow mask
    pillow_image = pipe(image) # applies mask on input and returns a pillow image
    st.image(pillow_image, caption='Processed Image.', use_column_width=True)