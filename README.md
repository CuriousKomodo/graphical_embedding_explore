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
will only modify the `n_neighbors` parameter in ConfigMyKNNClassifier

Experiment Configuration has to be loaded at the start of your script, as shown in `train_my_model.py`

```bash
python train_my_model.py --experiment-config ./experiments/experiment_1.yaml
```

# Trains

## trains.conf

The instance you are running trains on should have a trains.conf file installed  
under `/home/ubuntu/trains.conf`

## PyCharm plugin

To use Trains with PyCharm you need to have the PyCharm plugin installed.
You can find it here:  https://github.com/allegroai/trains-pycharm-plugin/releases

The plugin configuration is stored in the `dev/trains` secret in the AWS Secrets Manager
