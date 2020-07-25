# importing neptune
import neptune
# File data version
from neptunecontrib.versioning.data import log_data_version

# set UserName/ExperimentName
neptune.init('User/House_Price')

# set Train/Test file path
TRAIN_FILEPATH = '/demo/data/train.csv'
TEST_FILEPATH = '/demo/data/test.csv'
with neptune.create_experiment():
    log_data_version(TRAIN_FILEPATH)
    log_data_version(TEST_FILEPATH)

