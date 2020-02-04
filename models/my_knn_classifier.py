import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


from config.my_knn_classifier import ConfigMyKNNClassifier

module_logger = logging.getLogger('main_app.{}'.format(__file__))


class MyKNNClassifier:

    def __init__(self, config=None):

        self.config = config or ConfigMyKNNClassifier()
        module_logger.info('Configuration loaded: {}'.format(self.config))

        self.model = neighbors.KNeighborsClassifier(n_neighbors=self.config.n_neighbors,
                                                    weights=self.config.weights)

    def train(self, data, labels):
        module_logger.info('Training model ...')
        self.model.fit(data, labels)

    def predict(self, data):
        module_logger.info('Predicting model ...')
        return self.model.predict(data)

    def plot_predictions(self, data, predictions):
        plt.figure()
        plt.xlim(data[:, 0].min(), data[:, 0].max())
        plt.ylim(data[:, 1].min(), data[:, 1].max())
        plt.scatter(data[:, 0], data[:, 1], c=predictions)
        plt.title('3-Class classification (k = {}, weights = {})'.format(self.config.n_neighbors, self.config.weights))
        plt.show()
