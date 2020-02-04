import os
import logging

import matplotlib.pyplot as plt
from sklearn import neighbors
from natebbcommon.io_funcs import pickle_save, pickle_load

from config.my_knn_classifier import ConfigMyKNNClassifier

module_logger = logging.getLogger('main_app.models.my_knn_classifier')


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

    def save(self, output_directory):
        pickle_save(os.path.join(output_directory, 'model.pickle'), self.model)
        self.config.save(output_directory)
        module_logger.info('Saved model in {}'.format(output_directory))

    def load(self, output_directory):
        self.model = pickle_load(os.path.join(output_directory, 'model.pickle'))
        self.config = ConfigMyKNNClassifier().load(output_directory)
        module_logger.info('Loaded model from {}'.format(output_directory))
