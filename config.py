MODEL_FILE = "resnet50_model.h5"
HISTORY_FILE = "history.npz"
CM_FILE = "confusion_matrix.pkl"
TEST_DATA_FILE = "test_data.npz"
METRICS_FILE = "metrics.csv"
ACCURACY_CLASS_FILE = "class_accuracy.npz"
TEST_METRICS_FILE = "test_metrics.npz"


MODEL_PATH = "model/" + MODEL_FILE
HISTORY_PATH = "model/" + HISTORY_FILE
CM_PATH = "model/" + CM_FILE
TEST_DATA_PATH = "model/" +TEST_DATA_FILE
METRICS_PATH = "model/" + METRICS_FILE
METRICS_URL = "https://huggingface.co/sultanmr/cifar10_resnet50_classifier/blob/main/evaluation/metrics.csv"

ACCURACY_CLASS_PATH ="model/" + ACCURACY_CLASS_FILE
TEST_METRICS_PATH = "model/" + TEST_METRICS_FILE

MODEL_NAME = "cifar10_resnet50_classifier"
REPO_ID = "sultanmr/cifar10_resnet50_classifier"