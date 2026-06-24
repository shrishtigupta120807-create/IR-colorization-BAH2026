import streamlit as st
from PIL import Image
from pathlib import Path


st.set_page_config(
    page_title="IR Image Colorization Demo",
    layout="wide"
)

st.title("IR Image Colorization and Evaluation Demo")

st.write(
    "This demo shows the infrared input image, predicted RGB output, "
    "ground truth RGB image, and evaluation metrics."
)
uploaded_file = st.file_uploader(
    "Upload a TIR image for preview",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.subheader("Uploaded TIR Preview")
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, use_container_width=True)

BASE_DIR = Path(__file__).resolve().parent.parent

tir_path = BASE_DIR / "output/patches/demo/sample_006/tir_100m_512.png"
ground_truth_path = BASE_DIR / "output/patches/demo/sample_006/rgb_100m_512.png"
predicted_path = BASE_DIR / "member3_work/prediction.png"
result_path = BASE_DIR / "member4_work/evaluation_result.txt"

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Input TIR Image")
    st.image(Image.open(tir_path), use_container_width=True)

with col2:
    st.subheader("Predicted RGB Image")
    st.image(Image.open(predicted_path), use_container_width=True)

with col3:
    st.subheader("Ground Truth RGB Image")
    st.image(Image.open(ground_truth_path), use_container_width=True)

st.subheader("Evaluation Results")

if result_path.exists():
    st.text(result_path.read_text())
else:
    st.warning("evaluation_result.txt not found. Run the evaluation script first.")