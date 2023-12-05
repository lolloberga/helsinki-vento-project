import yaml
from typing import Any, Dict

import os

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
).replace('/config', '')


class ConfigParser:

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(ROOT_DIR, 'resources', 'configurations.yml')
        with open(config_path, 'r') as yamlfile:
            self._config = yaml.unsafe_load(yamlfile)
            self._define_constants()

    def __getitem__(self, name) -> Any:
        return self._config[name]

    @property
    def consts(self) -> Dict:
        return self._consts

    def _define_constants(self) -> None:
        self._consts = dict()

        self._consts['MAIN_PATH'] = ROOT_DIR
        self._consts['RESOURCE_PATH'] = os.path.join(ROOT_DIR, self._config['resouces']['path']) if \
            self._config['resouces']['path'] is not None else os.path.join(ROOT_DIR, "resources")
        self._consts['DATASET_PATH'] = os.path.join(ROOT_DIR, self._config['dataset']['path']) if \
            self._config['dataset']['path'] is not None else os.path.join(self._consts['RESOURCE_PATH'], "dataset")
        self._consts['MODEL_DRAWS_PATH'] = os.path.join(ROOT_DIR, self._config['model']['draws_path']) if \
            self._config['model']['draws_path'] is not None else os.path.join(ROOT_DIR, "../model", "draws")
        self._consts['MODEL_CHECKPOINT_PATH'] = os.path.join(ROOT_DIR, self._config['model']['checkpoint_path']) if \
            self._config['model']['checkpoint_path'] is not None else os.path.join(ROOT_DIR, "../model", "checkpoints")
