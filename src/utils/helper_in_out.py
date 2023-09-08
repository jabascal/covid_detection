import re

import yaml
    
def load_config(config_file):
    # Import YAML parameters from config/config.yaml

    # define custom tag handler
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def get_loader():
        # Add constructors to the loader
        loader = yaml.SafeLoader
        loader.add_constructor('!join', join)
        return loader

    with open(config_file, 'r') as stream:        
        param = yaml.load(stream, Loader=get_loader())
        print(yaml.dump(param))
    return param
