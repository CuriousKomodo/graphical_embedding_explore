import os
import argparse
import logging
import yaml

from trains import Task
from sklearn import datasets
from natebbcommon.config.experiment import ConfigExperiment
from natebbcommon.logger import initialise_logger

from config.my_knn_classifier import ConfigMyKNNClassifier
from models.my_knn_classifier import MyKNNClassifier

module_logger = logging.getLogger('main_app.{}'.format(__file__))


def main():
    task = Task.init(project_name="nate.blackbox.base", task_name="Test 1",
                     output_uri='/home/ubuntu/data_store/trains/snapshots')

    task.connect_configuration(ConfigExperiment.config)

    # import some data to play with
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target

    sample_model = MyKNNClassifier()
    sample_model.train(data=x, labels=y)
    predicted_labels = sample_model.predict(x)
    sample_model.plot_predictions(x, predicted_labels)
    module_logger.info('Training finished !')


if __name__ == '__main__':
    root_logger = initialise_logger(log_file_path=os.path.join(ConfigMyKNNClassifier().output_path, 'output.log'),
                                    syslog_server='logs4.papertrailapp.com',
                                    syslog_port=49313,
                                    syslog_hostname='nate.blackbox.base')

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-config', default=None, type=str,
                        help='Experiment configuration file to load.')
    pargs = parser.parse_args()

    if pargs.experiment_config:
        with open(pargs.experiment_config, 'r') as yaml_file:
            ConfigExperiment.config = yaml.load(yaml_file)
            module_logger.info('Loaded experiment configuration: {}'.format(ConfigExperiment.config))

    main()
