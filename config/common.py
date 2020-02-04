import os

from natebbcommon.config.base import ConfigBase


class ConfigCommon(ConfigBase):
    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    TRAINING_DATA_DIR = '/home/ubuntu/data_store/training_data/nate.blackbox.base'
    TRAINED_MODEL_DIR = '/home/ubuntu/data_store/trained_models/nate.blackbox.base'

