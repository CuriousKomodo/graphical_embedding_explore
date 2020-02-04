import argparse
import logging

import yaml
import numpy as np
from trains import Task
from sklearn import datasets
from natebbcommon.config.experiment import ConfigExperiment
from natebbcommon.logger import initialise_logger
from natebbcommon.io_funcs import create_directory_path_with_timestamp

from config.common import ConfigCommon
from models.my_knn_classifier import MyKNNClassifier

module_logger = logging.getLogger('main_app.train_my_model')


def main():
    task = Task.init(project_name="nate.blackbox.base", task_name="Test 1",
                     output_uri='/home/ubuntu/data_store/trains/snapshots')

    task.connect_configuration(ConfigExperiment.config)

    output_path = create_directory_path_with_timestamp(ConfigCommon().TRAINED_MODEL_DIR)

    # import some data to play with
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target

    sample_model = MyKNNClassifier()
    sample_model.train(data=x, labels=y)
    predicted_labels = sample_model.predict(x)
    sample_model.plot_predictions(x, predicted_labels)
    sample_model.save(output_path)

    module_logger.info('Loading model..')
    sample_model_loaded = MyKNNClassifier()
    sample_model_loaded.load(output_path)
    predicted_labels_loaded = sample_model_loaded.predict(x)

    assert np.array_equal(predicted_labels,predicted_labels_loaded) is True
    module_logger.info('Finished !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-config', default=None, type=str,
                        help='Experiment configuration file to load.')
    pargs = parser.parse_args()

    if pargs.experiment_config:
        with open(pargs.experiment_config, 'r') as yaml_file:
            ConfigExperiment.config = yaml.load(yaml_file, Loader=yaml.FullLoader)
            module_logger.info('Loaded experiment configuration: {}'.format(ConfigExperiment.config))

    root_logger = initialise_logger(syslog_server='logs4.papertrailapp.com',
                                    syslog_port=49313,
                                    syslog_hostname='nate.blackbox.base')

    main()
