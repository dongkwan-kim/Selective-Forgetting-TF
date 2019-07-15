from copy import deepcopy
from pprint import pprint
from typing import Dict, Callable
import os

from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
from termcolor import cprint

from utils import get_project_dir
import hashlib


class MyParams(HParams):
    """
    Inspired from:
    https://hanxiao.github.io/2017/12/21/Use-HParams-and-YAML-to-Better-Manage-Hyperparameters-in-Tensorflow/
    """
    def __init__(self,
                 yaml_file_to_config_name: dict,
                 value_magician: Dict[str, Callable] = None,
                 update_type="UNIQUE_OR_ERROR"):
        super().__init__()

        self.update_type = update_type
        self.yaml_file_to_config_name = yaml_file_to_config_name

        for yaml_file, config_name in yaml_file_to_config_name.items():

            with open(yaml_file) as fp:
                for k, v in YAML().load(fp)[config_name].items():
                    self.add_hparam_by_type(k, v)

        if value_magician is not None:
            self.run_magic(value_magician)

        self.param_hash = self.create_hash()

    def add_hparam_by_type(self, k, v):

        if v == "None":
            v = None

        if self.update_type == "UNIQUE_OR_ERROR":
            assert k not in self.values(), "{} cannot be updated at UNIQUE condition".format(k)
            self.add_hparam(k, v)
        elif self.update_type == "FIRST_PRIVILEGE":
            if k not in self.values():
                self.add_hparam(k, v)
        elif self.update_type == "LAST_PRIVILEGE":
            if k not in self.values():
                self.add_hparam(k, v)
            else:
                self.set_hparam(k, v)
        else:
            raise ValueError("{} is not appropriate update_type".format(self.update_type))

    def run_magic(self, value_magician):
        copied_self = deepcopy(self)
        for k in self.values():
            if k in value_magician:
                changed_v = value_magician[k](copied_self)
                self.del_hparam(k)
                self.add_hparam(k, changed_v)

    def pprint(self):
        for k, v in self.yaml_file_to_config_name.items():
            cprint("{} from {}".format(v, k), "green")
        pprint(self.values())
        cprint("hash: {}".format(self.get_hash()), "green")

    def has(self, k: str):
        return k in self

    def create_hash(self):
        strings = "/ ".join(["{}: {}".format(k, self.get(k)) for k in self.values()])
        return hashlib.md5(strings.encode()).hexdigest()

    def get_hash(self):
        return self.param_hash


def check_params(params):
    assert params.mtype in params.support_model, \
        "{} is not in {}".format(params.mtype, params.support_model)

    cprint("Params checked", "green")


def to_yaml_path(yaml_name):
    yaml_dir = os.path.join(get_project_dir(), "yamls")
    return os.path.join(yaml_dir, yaml_name)


if __name__ == '__main__':

    my_params = MyParams({
        to_yaml_path("experiment.yaml"): "SFDEN_FORGET",
        to_yaml_path("models.yaml"): "SMALL_FC_MNIST",
    }, value_magician={
        "checkpoint_dir": lambda p: os.path.join(
            p.checkpoint_dir, p.model, p.mtype,
        ),  # p is MyParams instance (== self)
    })

    # noinspection PyTypeChecker
    check_params(my_params)
    my_params.pprint()
