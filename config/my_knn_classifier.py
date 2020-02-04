from natebbcommon.config.base import ConfigBase
from natebbcommon.io_funcs import create_directory_path_with_timestamp

from config.common import ConfigCommon


class ConfigMyKNNClassifier(ConfigBase):
    output_path = create_directory_path_with_timestamp(ConfigCommon().TRAINED_MODEL_DIR)
    n_neighbors = 15
    weights = 'uniform'
