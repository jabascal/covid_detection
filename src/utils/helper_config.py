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

def set_params_debug(param):
    param['mlflow']['experiment_id'] = None
    param['mlflow']['experiment_name'] = "debug"
    #
    param['data']['cache'] = False
    param['data']['train_size'] = 2
    param['data']['val_size'] = 2
    param['data']['test_size'] = 2    
    param['data']['img_height'] = 64
    param['data']['img_width'] = 64
    #
    param['train']['epochs'] = 15    
    param['train']['batch_size'] = 16    
    return param
