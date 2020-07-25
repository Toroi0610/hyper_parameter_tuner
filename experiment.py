# importing neptune
import neptune
# File data version
from neptunecontrib.versioning.data import log_data_version
# File Loader
from data_loader import data_loader
# Hyperparameter Tuner
from hyperparameter_optimizer import create_objective
# Model
from model.lightgbm_0.model import learning


# set UserName/ExperimentName
neptune.init('User/House_Price')

# set Train/Test file path
TRAIN_FILEPATH = '/data/train.csv'
TEST_FILEPATH = '/data/test.csv'
TARGET_LIST = ["SalePrice"]
with neptune.create_experiment():
    log_data_version(TRAIN_FILEPATH)
    log_data_version(TEST_FILEPATH)

training_data, training_target = data_loader(TRAIN_FILEPATH, TARGET_LIST)
test_data, _ = data_loader(TEST_FILEPATH)

neptune.create_experiment('House_Price')
neptune_callback = optuna_utils.NeptuneCallback()

objective = lambda trail : create_objective(trail, learning, training_data, target, metric, validation_size=0.25)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
optuna_utils.log_study(study)