
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import numpy as np


def load_keras_model():
    # Directly load model from Hugging Face
    model_path = hf_hub_download(
        repo_id="sultanmr/cifar10_resnet50_classifier",
        filename="models/resnet50_model.h5",  # Path in your repo
        repo_type="model"
    )
    return load_model(model_path)

model = load_keras_model()
print(model.summary())




'''
from transformers import push_to_hub_keras
from tensorflow.keras.models import load_model
from config import *

model = load_model(MODEL_PATH)  # Your trained Keras model
push_to_hub_keras(model, "sultanmr/cifar10_resnet50_classifier")

from transformers import TFAutoModelForImageClassification
model = TFAutoModelForImageClassification.from_pretrained("sultanmr/cifar10_resnet50_classifier")
print (model.summary())

from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="sultanmr/cifar10_resnet50_classifier",
    filename="resnet50_model.h5"
)

# Load with Keras
model = load_model(model_path)
print (model.summary())  # Now this will work!


from huggingface_hub import model_info
print(model_info("sultanmr/cifar10_resnet50_classifier"))
'''