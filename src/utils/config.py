import yaml

def load_params(params_path: str = "params.yaml"):
    """Load parameters from YAML file"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params
