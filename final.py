# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from typing import Tuple
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from datetime import datetime
from chatbot_final import create_chatbot
from chatbot_final import ask_question

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Medical Segmentation & Report Generator",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# Config - Update model paths here
# ------------------------------
IMAGE_SIZE = (256, 256)
CLASSIFIER_MODEL_PATH = "modality_classifier.keras"
XRAY_MODEL_PATH = "unet_pneumo_grayscale_5000.keras"
CT_MODEL_PATH   = "unet_brain_tumor_grayscale.keras"

# ------------------------------
# Load Models (cached)
# ------------------------------
@st.cache_resource
def load_keras_model(path: str):
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception:
        return None

@st.cache_resource
def load_all_models():
    classifier = load_keras_model(CLASSIFIER_MODEL_PATH)
    xray_model  = load_keras_model(XRAY_MODEL_PATH)
    ct_model    = load_keras_model(CT_MODEL_PATH)
    return classifier, xray_model, ct_model

classifier_model, xray_model, ct_model = load_all_models()

# ------------------------------
# Preprocess helper (grayscale)
# ------------------------------
def preprocess_image_from_bytes(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize(IMAGE_SIZE)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=-1)
    batch = np.expand_dims(img_np, axis=0)
    return batch.astype(np.float32), img_np.astype(np.float32)

# ------------------------------
# Prediction & postprocess
# ------------------------------
def predict_segmentation(model, batch_img: np.ndarray, threshold: float=0.5):
    pred = model.predict(batch_img, verbose=0)
    pred_mask = pred[0, :, :, 0]
    binary_mask = (pred_mask > threshold).astype(np.float32)
    pct = (np.sum(binary_mask) / binary_mask.size) * 100.0
    return pred_mask, binary_mask, pct, int(np.sum(binary_mask)), int(binary_mask.size)

# ------------------------------
# Visualization creator
# ------------------------------
def create_visualization(img_np: np.ndarray, pred_mask: np.ndarray, binary_mask: np.ndarray, percent: float):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_np[:, :, 0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability Map')
    axes[1].axis('off')

    axes[2].imshow(binary_mask, cmap='Reds')
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')

    axes[3].imshow(img_np[:, :, 0], cmap='gray')
    axes[3].imshow(binary_mask, cmap='Reds', alpha=0.6)
    axes[3].set_title(f'Overlay\nCoverage: {percent:.2f}%')
    axes[3].axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ------------------------------
# PDF report generator
# ------------------------------
try:
    from reportlab.lib.utils import ImageReader
except Exception:
    ImageReader = None

def generate_pdf_report(patient_info: dict,
                        modality: str,
                        diagnosis_label: str,
                        tumor_pct: float,
                        tumor_pixels: int,
                        total_pixels: int,
                        notes: str,
                        viz_png_bytes: bytes) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    left_margin = 20 * mm
    top = height - 20 * mm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, top, "Medical Segmentation Report")
    c.setFont("Helvetica", 10)
    c.drawString(left_margin, top - 12, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    y = top - 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Patient Details:")
    c.setFont("Helvetica", 10)
    y -= 14
    c.drawString(left_margin, y, f"Name: {patient_info.get('name', '-')}")
    y -= 12
    c.drawString(left_margin, y, f"Age: {patient_info.get('age', '-')}")
    y -= 12
    c.drawString(left_margin, y, f"Blood Group: {patient_info.get('blood_group', '-')}")
    y -= 12
    c.drawString(left_margin, y, f"Contact: {patient_info.get('contact', '-')}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Study & Results:")
    c.setFont("Helvetica", 10)
    y -= 14
    c.drawString(left_margin, y, f"Modality: {modality.upper()}")
    y -= 12
    c.drawString(left_margin, y, f"Predicted Condition: {diagnosis_label}")
    y -= 12
    c.drawString(left_margin, y, f"Tumor/Coverage: {tumor_pct:.2f}% ({tumor_pixels} / {total_pixels} pixels)")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Clinical Notes:")
    c.setFont("Helvetica", 10)
    y -= 14
    text = c.beginText(left_margin, y)
    text.setFont("Helvetica", 10)
    for line in notes.splitlines():
        text.textLine(line)
    c.drawText(text)
    y -= (14 * (len(notes.splitlines()) + 1))

    precautions = []
    if modality == "xray":
        precautions = [
            "Pneumothorax precautions:",
            "- Seek immediate medical attention if sudden chest pain or breathlessness occurs.",
            "- Avoid heavy exertion until cleared by a clinician.",
            "- Follow-up chest X-ray may be recommended."
        ]
    else:
        precautions = [
            "CT scan (brain lesion) precautions:",
            "- Consult a neurologist for confirmation.",
            "- Avoid strenuous activity until advised.",
            "- Follow up with imaging as directed by your clinician."
        ]

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Precautions:")
    c.setFont("Helvetica", 10)
    y -= 14
    text = c.beginText(left_margin, y)
    text.setFont("Helvetica", 10)
    for line in precautions:
        text.textLine(line)
    c.drawText(text)

    y_img = y - (14 * (len(precautions) + 1)) - 10
    try:
        if viz_png_bytes and ImageReader:
            img_buf = io.BytesIO(viz_png_bytes)
            img = Image.open(img_buf)
            max_w = width - 2 * left_margin
            aspect = img.height / img.width
            img_w = max_w
            img_h = max_w * aspect
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_buf2 = io.BytesIO()
            img.save(img_buf2, format="PNG")
            img_buf2.seek(0)
            c.drawImage(ImageReader(img_buf2), left_margin, y_img - img_h, width=img_w, height=img_h)
    except Exception:
        pass

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(left_margin, 10 * mm, "This is an automated report for informational purposes only.")
    c.save()
    buf.seek(0)
    return buf

# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.title("🩺 Multi-Model Segmentation & Report Generator")
    st.markdown("Upload an image (X-ray or CT Scan). The app will segment and generate a PDF report.")

    with st.sidebar:
        st.header("Patient Details")
        name = st.text_input("Name")
        age = st.text_input("Age")
        blood_group = st.text_input("Blood Group")
        contact = st.text_input("Contact")
        notes = st.text_area("Additional Notes", value="No additional notes.")

        st.divider()
        st.header("Segmentation Settings")
        confidence_threshold = st.slider(
            "Segmentation Threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )

        st.divider()
        st.header("Model Status")
        st.write(f"X-ray model loaded: {'✅' if xray_model is not None else '❌'}")
        st.write(f"CT Scan model loaded: {'✅' if ct_model is not None else '❌'}")

    # ✅ MODALITY SELECTED FIRST (ONLY CHANGE)
    st.subheader("Select Modality")
    modality = st.radio("Select Modality:", ("X-ray (Pneumothorax)", "CT Scan (Brain Tumor)"))
    modality = "xray" if "X-ray" in modality else "ct"

    if modality == "xray":
        seg_model = xray_model
        diag_label = "Pneumothorax (suspected)"
    else:
        seg_model = ct_model
        diag_label = "Brain tumor / lesion (suspected)"

    if seg_model is None:
        st.error("Segmentation model for selected modality is not available.")
        return

    # ⬇️ IMAGE UPLOAD COMES AFTER MODALITY
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        f"Upload {modality.upper()} image",
        type=["png", "jpg", "jpeg", "tiff", "bmp"]
    )

    if uploaded_file is None:
        st.info("Please upload an image to continue.")
        return

    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    st.subheader("Uploaded Image Preview")
    st.image(file_bytes, use_column_width=True)

    with st.spinner("Preprocessing..."):
        try:
            batch_img, img_np = preprocess_image_from_bytes(file_bytes)
            st.success("Image preprocessed successfully.")
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return

    with st.spinner("Running segmentation..."):
        try:
            pred_mask, binary_mask, pct, mask_pixels, total_pixels = predict_segmentation(
                seg_model, batch_img, threshold=confidence_threshold
            )
            st.success("Segmentation completed successfully.")
        except Exception as e:
            st.error(f"Error during segmentation: {e}")
            return

    st.subheader("Results")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Coverage", f"{pct:.2f}%")
        st.metric("Mask Pixels", f"{mask_pixels:,}")
    with col2:
        st.metric("Total Pixels", f"{total_pixels:,}")

    if modality == "xray":
        if pct > 0.4:
            st.error("⚠️ Pneumothorax suspected — urgent evaluation recommended.")
        else:
            st.success("✅ No significant pneumothorax detected.")
    else:
        if pct > 0.4:
            st.warning("⚠️ Lesion/tumor detected — further evaluation advised.")
        else:
            st.success("✅ No significant lesion detected.")

    st.subheader("Segmentation Visualization")
    viz_buf = create_visualization(img_np, pred_mask, binary_mask, pct)
    st.image(viz_buf, use_column_width=True)

    st.download_button(
        "Download Visualization PNG",
        data=viz_buf,
        file_name="visualization.png",
        mime="image/png"
    )

    mask_img = (binary_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img)
    mask_buf = io.BytesIO()
    mask_pil.save(mask_buf, format="PNG")
    mask_buf.seek(0)

    st.download_button(
        "Download Binary Mask PNG",
        data=mask_buf,
        file_name="mask.png",
        mime="image/png"
    )

    st.subheader("Generate PDF Report")
    if st.button("Create & Download PDF Report"):
        with st.spinner("Generating PDF..."):
            patient_info = {
                "name": name or "-",
                "age": age or "-",
                "blood_group": blood_group or "-",
                "contact": contact or "-"
            }
            try:
                viz_bytes = viz_buf.getvalue()
                pdf_buf = generate_pdf_report(
                    patient_info=patient_info,
                    modality=modality,
                    diagnosis_label=diag_label,
                    tumor_pct=pct,
                    tumor_pixels=mask_pixels,
                    total_pixels=total_pixels,
                    notes=notes,
                    viz_png_bytes=viz_bytes
                )
                st.success("PDF report generated successfully.")
                st.download_button(
                    "Download PDF Report",
                    data=pdf_buf,
                    file_name="medical_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

if __name__ == "__main__":
    main()

st.divider()
st.sidebar.header("💬 Disease Chatbot")

# Load chatbot only once
PDF_PATHS = ["pneumo pdf.pdf", "PE-Brain-tumors_UCNI.pdf"]

rag_chain, llm, db = create_chatbot(PDF_PATHS)

st.divider()
st.sidebar.header("💬 Disease Chatbot")

# Sidebar chatbot input
user_question = st.sidebar.text_input("Ask about Pneumothorax or Brain Tumor:")

if user_question:
    with st.spinner("Fetching answer..."):
        try:
            answer = ask_question(user_question, rag_chain, llm, db)
            st.sidebar.success(answer)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
