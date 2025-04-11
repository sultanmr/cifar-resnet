import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import mlflow
import datetime
from config import *

# CIFAR-10 labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = load_model(MODEL_PATH)

st.title("🖼️ Simple CIFAR-10 Image Predictor for " + ", ".join(labels))
st.markdown(f"[🔗 View MLflow Dashboard]({MLFLOW_MODEL_URI})")

# Upload and predict
uploaded = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"uploaded_{timestamp}.png"
    img_path = os.path.join("tmp", img_filename)
    os.makedirs("tmp", exist_ok=True)

    with open(img_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Show spinner while processing
    with st.spinner('Processing your image...'):
        # Resize and preprocess
        img = load_img(img_path, target_size=(32, 32))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        pred_label = labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

    st.success(f"Predicted Label: **{pred_label}** with confidence {confidence:.2%}")

    # Save prediction result
    result_path = os.path.join("tmp", f"result_{timestamp}.txt")
    with open(result_path, "w") as f:
        f.write(f"Predicted Label: {pred_label}\nConfidence: {confidence:.2%}")

    # Log to MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"upload_{timestamp}"):
        mlflow.log_param("predicted_label", pred_label)
        mlflow.log_metric("confidence", confidence)
        mlflow.log_artifact(img_path, artifact_path="uploaded_images")
        mlflow.log_artifact(result_path, artifact_path="prediction_results")
