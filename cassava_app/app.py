import streamlit as st
from inference_sdk import InferenceHTTPClient
import requests
from PIL import Image
import tempfile
import os
import cv2
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Cassava Disease Detection", layout="centered")

# -----------------------------
# CONFIGURATION
# -----------------------------
ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

MODEL_ID = "cassavadisease/1"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"

# -----------------------------
# INITIALIZE ROBOFLOW CLIENT
# -----------------------------
CLIENT = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY
)

# -----------------------------
# FUNCTION: AI EXPLANATION
# -----------------------------
def get_ai_explanation(disease_name):

    prompt = f"""
    Explain briefly the cassava disease: {disease_name}.
    Include:
    - Cause
    - Main Symptoms
    - Prevention
    - Treatment
    Keep answer short.
    """

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Cassava Disease Detection App"
        },
        json={
            "model": "minimax/minimax-m2.5",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.3
        }
    )

    if response.status_code != 200:
        return f"OpenRouter API Error:\n{response.text}"

    result = response.json()

    if "choices" not in result:
        return f"Unexpected API Response:\n{result}"

    return result["choices"][0]["message"]["content"]


# -----------------------------
# UI
# -----------------------------
st.title("Cassava Disease Detection Web App")
st.write("Upload or capture a cassava leaf image for disease detection.")

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

# -----------------------------
# MAIN PROCESS
# -----------------------------
if image is not None:

    st.image(image, caption="Captured Image", use_container_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Roboflow inference
    with st.spinner("Analyzing image..."):
        result = CLIENT.infer(temp_path, model_id=MODEL_ID)

    os.remove(temp_path)

    predictions = result.get("predictions", [])

    if predictions:

        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        for pred in predictions:

            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            label = pred["class"]
            confidence = round(pred["confidence"] * 100, 2)

            # Convert center to corner format
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            cv2.rectangle(img_cv, (x1, y1 - 30), (x1 + 250, y1), (0, 255, 0), -1)

            # Label text
            cv2.putText(
                img_cv,
                f"{label} ({confidence}%)",
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        # Convert back to RGB
        img_display = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(img_display, caption="Detected & Labeled Image", use_container_width=True)

        # Get highest confidence prediction
        top_prediction = max(predictions, key=lambda x: x["confidence"])
        disease_name = top_prediction["class"]
        confidence = round(top_prediction["confidence"] * 100, 2)

        st.success(f"Detected: **{disease_name}**")
        st.info(f"Confidence: {confidence}%")

        # AI Explanation
        with st.spinner("Generating disease explanation..."):
            explanation = get_ai_explanation(disease_name)

        st.markdown("## 📘 Disease Information")
        st.write(explanation)

    else:
        st.warning("No cassava leaf detected.")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>"
    "Developed by <b>Edcel Bogay</b>"
    "</div>",
    unsafe_allow_html=True
)