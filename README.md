# nate.blackbox.base


# Experiments & Configurations

Default configurations for each model are stored in `config/<model_name>.py`

Experiment YAML files, defined in `experiments/<name>.yaml` overwrite the default configuration of these classes.
Their structure is the class name followed by only the parameters that need to be different versus the default:

For example this `experiment.yaml` file:
```yaml
ConfigMyKNNClassifier:
  n_neighbors : 20

ConfigMyMLPClassifier:
  epochs : 10
```
will only modify the `n_neighbors` parameter in ConfigMyKNNClassifier and 
`epochs` in ConfigMyMLPClassifier.

Experiment Configuration has to be loaded at the start of your script, as shown in `train_my_model.py`

```bash
python train_my_model.py --experiment-config ./experiments/experiment_1.yaml
```

# Trains

The only lines of code you will need at the start of your main script (e.g train model script) are:

```python
from trains import Task
from natebbcommon.config.experiment import ConfigExperiment

# load the experiment config here if needed

task = Task.init(project_name="nate.blackbox.base", task_name="Test 1",
                 output_uri='/home/ubuntu/data_store/trains/snapshots')

task.connect_configuration(ConfigExperiment.config)

# Do everything else afterwards
```

## trains.conf

The instance you are running trains on should have a trains.conf file installed under `/home/ubuntu/trains.conf`

If this file is absent then you can restore it using:

```bash
ln -s /home/ubuntu/data_store/trains/config_home/trains.conf /home/ubuntu/trains.conf
```

## PyCharm plugin

To use Trains with PyCharm you need to have the PyCharm plugin installed.
You can find it on their [pycharm plugin repository](https://github.com/allegroai/trains-pycharm-plugin/releases "github.com/allegroai")
and the instructions for installing it are [here](https://allegro.ai/docs/deploying_trains/trains_plugins/#trains-pycharm-plugin "allegro.ai/docs").

The plugin configuration is stored in the [dev/trains](https://eu-west-1.console.aws.amazon.com/secretsmanager/home?region=eu-west-1#/secret?name=dev%2Ftrains) secret in the AWS Secrets Manager
