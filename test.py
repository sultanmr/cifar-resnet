from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import *



model = load_model(MODEL_PATH)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load test data
data = np.load(TEST_DATA_PATH)
x_test = data['images'] 
y_test = data['labels']  

plot_model(model, to_file='model_architecture.png', 
           show_shapes=True, 
           show_layer_names=True,
           expand_nested=True)

# Get predictions
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=-1).astype(int)

# write y_test and y_pred_labels to a csv file
y_test_labels = np.argmax(y_test, axis=-1).astype(int) 

#np.savetxt("y_test_labels.csv", y_test_labels, delimiter=",")
#np.savetxt("y_pred_labels.csv", y_pred_labels, delimiter=",")

cm = confusion_matrix(y_test_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
# Save confusion matrix plot locally and log it to MLflow
plt.savefig("confusion_matrix.png")

    