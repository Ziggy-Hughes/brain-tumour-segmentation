import os
import yaml

config_fname = 'config.yml'
folder_path = os.path.abspath('.')
config_fpath = os.path.join(folder_path, config_fname)
_config = {}

def load_configuration():
    print("Loading configuration")
    try:
        with open(config_fpath, "r") as f:
            read_config = yaml.safe_load(f)
            return read_config
    except Exception as e:
        print("Reading configuration file {} failure: {}".format(config_fpath, e))
        raise
        
_config = load_configuration()
