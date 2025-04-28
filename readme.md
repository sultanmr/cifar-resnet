# Cifar-Resnet Image Detection
[Live Demo](https://cifar-resnet-jmydtirbappnr8cgextdtag.streamlit.app/)

![image](https://github.com/user-attachments/assets/15da1f91-9be1-4727-ab8f-a3d25ce59dcc)

           
This repository contains a deep learning project that utilizes the ResNet50 model for classifying images from the CIFAR-10 dataset. The model was trained on a subset of the CIFAR-10 dataset, and various tools, including MLflow and Streamlit, were used to track the experiment and deploy a web-based UI.
git a
## Project Overview

- **Dataset**: CIFAR-10, a dataset containing 60,000 32x32 color images in 10 classes.
- **Model**: A modified ResNet50 architecture, where the top layers of the model are replaced with custom dense layers for classification.
- **Tools Used**:
  - **TensorFlow/Keras** for building and training the model.
  - **MLflow** for experiment tracking and model versioning.
  - **Streamlit** for creating a simple web app to predict CIFAR-10 images.

## Project Components

### 1. Model Training (Google Colab)
[Colab Link](https://github.com/sultanmr/cifar-resnet/blob/main/train_cifar10_resnet50.ipynb)
The model training code was run on Google Colab, where the following steps were carried out:
- Loaded the CIFAR-10 dataset and preprocessed the images and labels.
- Used a pre-trained ResNet50 model (without the top layers) and fine-tuned it for classification.
- The model was saved as `resnet50_model.h5`.

For details on the training process and how the model was built, refer to the Colab notebook you can find in the repository.

### 2. Web Application (Streamlit)

A simple Streamlit web app (`app.py`) was created to allow users to upload an image, which will then be classified using the trained model. The app displays the predicted class and confidence score of the image.

You can run the app with the following command:

```bash
streamlit run app.py
```

### 3. Experiment Tracking with MLflow

The training and prediction results, including model parameters and metrics, are tracked using **MLflow**. You can view the results in the [MLflow Dashboard](https://dagshub.com/sultanmr/my-first-repo.mlflow/#/experiments/2/runs/d5b1592339de4915ab846b7e0bc41813/artifacts/).

### 4. Saving and Loading Models

The trained model (`resnet50_model.h5`) is saved for later use. You can download it and use it for inference with Streamlit or any other application.

###  5. Training Summary

- Test Accuracy: `0.7694`
- Test Loss: `0.7501`

- Best Validation Accuracy: `0.7711` (Epoch `40`)
- Final Training Accuracy: `0.8004`
- Final Validation Accuracy: `0.7711`
- Overfitting Gap: `0.0293`

After training, the final accuracy on the test dataset is printed. Additionally, the training history and test data are saved for further analysis and visualization.

![image](https://github.com/user-attachments/assets/4a3df249-c8aa-413f-9279-152c6afbc64d)
![image](https://github.com/user-attachments/assets/bc3da94d-3092-446f-adfc-8195a3224995)
![image](https://github.com/user-attachments/assets/5d2f82b5-e94a-4adb-aee8-13efc60c9ffc)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sultanmr/cifar-resnet.git
   cd cifar-resnet
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Access the MLflow Dashboard:
   Visit the following URL to view experiment metrics: [MLflow Dashboard](https://dagshub.com/sultanmr/my-first-repo.mlflow/#/experiments/2/runs/bfc550c5403b44c0a980c0629be2de58/artifacts)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This version of the README provides:
- An overview of the project.
- Instructions for installation and usage.
- A link to the MLflow dashboard.
- A reference to the code files for training and the web app, without including the actual code.
