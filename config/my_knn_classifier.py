from natebbcommon.config.base import ConfigBase


class ConfigMyKNNClassifier(ConfigBase):
    n_neighbors = 15
    weights = 'uniform'
