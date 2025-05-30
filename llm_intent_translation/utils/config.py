import yaml
from munch import Munch

def load_config(path):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return Munch(cfg_dict)
