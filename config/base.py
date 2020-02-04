`import os
import pprint
import logging


from natebbcommon.io_funcs import write_json_file, read_json_file

from config.experiment import ConfigExperiment

module_logger = logging.getLogger('main_app.config_base')


class ConfigBase:

    def __init__(self):
        self.config_class_name = self.__class__.__name__
        class_config = ConfigExperiment.config.get(self.config_class_name)
        if class_config:
            for key in class_config:
                setattr(self.__class__, key, class_config[key])

    def __repr__(self):
        return pprint.pformat(self.__class__.__dict__, indent=4)

    def __str__(self):
        return pprint.pformat(self.__class__.__dict__, indent=4)

    def save(self, destination_dir):
        destination_path = os.path.join(destination_dir, '{}.json'.format(self.config_class_name))
        write_json_file(destination_path, self.__class__.__dict__)
        module_logger.info('Configuration saved in {}'.format(destination_path))

    def load(self, destination_dir):
        destination_path = os.path.join(destination_dir, '{}.json'.format(self.config_class_name))
        self.__class__.__dict__ = read_json_file(destination_path)
        module_logger.info('Configuration loaded from {}'.format(destination_path))
