from natebbcommon.io_funcs import create_directory_path_with_timestamp
from natebbcommon.config.base import ConfigBase

from config.common import ConfigCommon


class ConfigMyKNNClassifier(ConfigBase):
    n_neighbors = 15
    weights = 'uniform'
    output_path = create_directory_path_with_timestamp(ConfigCommon().TRAINED_MODEL_DIR)
