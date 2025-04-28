import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import random
import pickle
from huggingface_hub import hf_hub_download
from PIL import Image
from config import *


@st.cache_resource  # Changed to cache_resource for models
def load_all_data():
    """Load all files in a single Hugging Face API call"""
    # Download all files at once
    files = {
        "model": hf_hub_download(repo_id=REPO_ID, filename=f"models/{MODEL_FILE}", repo_type="model"),
        "history": hf_hub_download(repo_id=REPO_ID, filename=f"evaluation/{HISTORY_FILE}", repo_type="model"),
        "test_metrics": hf_hub_download(repo_id=REPO_ID, filename=f"evaluation/{TEST_METRICS_FILE}", repo_type="model"),
        "cm": hf_hub_download(repo_id=REPO_ID, filename=f"evaluation/{CM_FILE}", repo_type="model"),
        "class_accuracy": hf_hub_download(repo_id=REPO_ID, filename=f"evaluation/{ACCURACY_CLASS_FILE}", repo_type="model")
    }
    
    # Process all data
    model = load_model(files["model"])
    
    history_data = np.load(files["history"])
    history_df = pd.DataFrame({key: history_data[key] for key in history_data.files})
    history_df["epoch"] = history_df.index + 1
    
    test_metrics = np.load(files["test_metrics"])
    test_loss, test_acc = test_metrics["test_loss"], test_metrics["test_acc"]
    
    with open(files["cm"], 'rb') as f:
        cm = pickle.load(f)
        
    class_accuracy = np.load(files["class_accuracy"])["class_accuracy"]
    
    (_, _), (test_images, test_labels) = cifar10.load_data()
    test_images = test_images.astype('float32') / 255.0        
    
    return {
        "model": model,
        "history": history_df,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "cm": cm,
        "class_accuracy": class_accuracy,
        "test_images": test_images,
        "test_labels": test_labels
    }
    
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


with st.spinner('Loading model and data...'):
    data = load_all_data()
    model = data["model"]
    history = data["history"]
    test_loss = data["test_loss"]
    test_acc = data["test_acc"]
    cm = data["cm"]
    class_accuracy = data["class_accuracy"]
    test_images = data["test_images"]
    test_labels = data["test_labels"]
    

# Sidebar menu styled like menu items
section = st.sidebar.radio("", ["Model Metrics", "Model Testing"])


if section == "Model Metrics":
    # --- Sidebar Metrics Section ---
    st.title("Model Metrics")  
    
    st.download_button(
        label="⬇️ Download metrics.csv",
        data=METRICS_URL,
        file_name="metrics.csv",
        mime="text/csv"
    )          

    # Training/Validation Metrics Chart

    # Sidebar to select metrics
    metrics = ["accuracy", "val_accuracy", "loss", "val_loss"]        
    
    

    st.markdown(f"Accuracy: {test_acc * 100:.2f}% | Loss: {test_loss:.4f}")

    selected_metrics = st.multiselect("Select metrics to display", metrics, default=metrics)
   
    # Line chart for Training/Validation Metrics
    if selected_metrics:
        st.line_chart(history[["epoch"] + selected_metrics].set_index("epoch"))
    else:
        st.info("Select at least one metric to view the chart.")


    col1, col2 = st.columns(2)
    with col1:
        # Plot the class-wise accuracy as a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(class_names, class_accuracy, color='skyblue')
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_title('Class-wise Accuracy')
        ax.set_ylim(0, 1)  # Set y-axis to range from 0 to 1 for better visualization
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=range(10))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        st.pyplot(plt)


elif section == "Model Testing":
    # --- Model Testing Section ---
    st.title("Model Testing")

    # Button to trigger random prediction
    if st.button("Pick random image to predict"):     
        random_idx = random.randint(0, len(test_images) - 1)
        img = test_images[random_idx]
        true_label = test_labels[random_idx] 
        # Create two columns to display the image and the prediction side by side
        col1, col2 = st.columns(2)
         # Display the image in the first column
        with col1:
            st.image(img, use_container_width=True)

        # Display the true and predicted class in the second column
        with col2:
            st.write(f"### True Label: {class_names[true_label[0]]}")
            # Show a loading spinner while the model is predicting
            result_placeholder = st.empty() 
            with st.spinner('Predicting... Please wait'):       
                
                pred = model.predict(np.expand_dims(img, axis=0))
                pred_class = np.argmax(pred, axis=1)
                pred_confidence = np.max(pred)             
            
            with result_placeholder.container():
                st.write(f"### Predicted {class_names[pred_class[0]]}")        
                st.write(f"### with {pred_confidence * 100:.2f}% confidence")

    

    # Button to upload an image
    uploaded_file = st.file_uploader("OR choose your image to predict...", type=["jpg", "png", "jpeg"])

    # If an image is uploaded, make a prediction
    if uploaded_file is not None:
        # Open the image and prepare it for prediction
        img = Image.open(uploaded_file)
         # Check if the image has an alpha channel and remove it if it does
        if img.mode == 'RGBA':
            img = img.convert('RGB')  # Convert RGBA to RGB (remove alpha channel)
        
        img = img.resize((32, 32))  # Resize to match CIFAR-10 image size (32x32)
        img = np.array(img).astype('float32') / 255.0
        
        if img.shape != (32, 32, 3):
            st.error(f"Image has an unexpected shape: {img.shape}. Expected shape is (32, 32, 3).")

        # Expand the dimensions to match the input shape of the model
        img = np.expand_dims(img, axis=0)

        # Show the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Show a loading spinner while the model is predicting
        with st.spinner('Predicting... Please wait'):
            pred = model.predict(img)
            pred_class = np.argmax(pred, axis=1)
            pred_confidence = np.max(pred)

        # Display the predicted class
        st.write(f"### Predicted {class_names[pred_class[0]]} with {pred_confidence * 100:.2f}% confidence")
