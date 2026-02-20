import streamlit as st
from roboflow import Roboflow
from PIL import Image
import requests
import numpy as np
import cv2
import tempfile
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Cassava Disease Detection", layout="centered")

# -----------------------------
# API KEYS
# -----------------------------
ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# -----------------------------
# INITIALIZE ROBOFLOW MODEL
# -----------------------------
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("cassavadisease")  # your project name
model = project.version(1).model  # version number

# -----------------------------
# FUNCTION: AI EXPLANATION
# -----------------------------
def get_ai_explanation(disease_name):
    prompt = f"""
    Explain briefly the cassava disease: {disease_name}.
    Include cause, symptoms, prevention, treatment.
    Keep answer short.
    """
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "minimax/minimax-m2.5",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.3
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# -----------------------------
# UI
# -----------------------------
st.title("Cassava Disease Detection Web App")
source = st.radio("Select Image Source:", ["Upload Image", "Use Camera"])
image = None

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif source == "Use Camera":
    camera_photo = st.camera_input("Take a picture of the cassava leaf")
    if camera_photo:
        image = Image.open(camera_photo)

if image:
    st.image(image, caption="Captured Image", use_container_width=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # --- Roboflow prediction ---
    with st.spinner("Analyzing image..."):
        result = model.predict(temp_path).json()
    os.remove(temp_path)

    predictions = result.get("predictions", [])

    if predictions:
        # Draw bounding boxes
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            label, conf = pred["class"], round(pred["confidence"]*100,2)
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img_cv, (x1, y1-30), (x1+250, y1), (0,255,0), -1)
            cv2.putText(img_cv, f"{label} ({conf}%)", (x1+5,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Detected & Labeled Image", use_container_width=True)

        # Show top prediction
        top_pred = max(predictions, key=lambda x: x["confidence"])
        st.success(f"Detected: {top_pred['class']} ({round(top_pred['confidence']*100,2)}%)")
        with st.spinner("Generating disease explanation..."):
            st.write(get_ai_explanation(top_pred["class"]))
    else:
        st.warning("No cassava leaf detected.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Developed by <b>Edcel Bogay</b></div>",
    unsafe_allow_html=True
)
