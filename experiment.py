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

# load api_token (Please register https://ui.neptune.ai/auth/realms/neptune/protocol/openid-connect/registrations?client_id=neptune-frontend&redirect_uri=https%3A%2F%2Fui.neptune.ai%2Fafter-registration&state=db8b7051-097e-44e5-83cc-ed156f15e995&response_mode=fragment&response_type=code&scope=openid&nonce=f2f5767b-e16b-47b9-8944-4fdff4d39f80)
with open("api_token.txt", "r") as f:
    api_token = f.read()
# set UserName/ExperimentName
neptune.init('User/House_Price',
             api_token=api_token)

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

objective = lambda trial : create_objective(trial, learning, training_data, training_target, metric, validation_size=0.25)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
optuna_utils.log_study(study)
