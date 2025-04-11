# Constants
MLFLOW_TRACKING_URI = "https://dagshub.com/sultanmr/my-first-repo.mlflow/"
MLFLOW_TRACKING_USERNAME = "sultanmr"
MLFLOW_TRACKING_PASSWORD = "79869daf4b3a7f1cdc2fffb3cd3d867c67454e2a"
MLFLOW_EXPERIMENT_NAME = 'cifar10-resnet50'
MLFLOW_MODEL_URI = "https://dagshub.com/sultanmr/my-first-repo.mlflow/#/experiments/2/runs/bfc550c5403b44c0a980c0629be2de58/artifacts"

MODEL_PATH = "model/resnet50_model.h5"
HISTORY_PATH = "model/history.npz"
TEST_DATA_PATH = "model/test_data.npz"
UPLOAD_LOG_DIR = "logged_uploads"



# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(10, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)