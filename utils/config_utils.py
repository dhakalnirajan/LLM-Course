import yaml
from typing import Dict, Any, Optional
from cerberus import Validator

config_schema = {
    'dataset_path': {'type': 'string', 'required': True},
    'batch_size': {'type': 'integer', 'required': True, 'min': 1},
    'learning_rate': {'type': 'float', 'required': True, 'min': 0},
}

class Config:
    """
    A class for managing configuration parameters from a YAML file.
    """
    def __init__(self, config_path: str):
        """
        Loads configuration from the specified YAML file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
    """Loads the YAML configuration file and validates against schema"""
    try:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {self.config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}

    v = Validator(config_schema)
    if not v.validate(config):
        print(f"Error validating config file: {v.errors}")
        return {}
    return v.document

    def get(self, key: str, default: Optional[Any] = None) -> Any:
    """
    Retrieves a configuration value, checking environment variables first.
    """
    env_var_name = key.upper()  # e.g., BATCH_SIZE
    env_var = os.environ.get(env_var_name)
    if env_var is not None:
        # Attempt to convert to appropriate type
        try:
            if isinstance(default, int):
                return int(env_var)
            elif isinstance(default, float):
                return float(env_var)
            elif isinstance(default, bool):
                return env_var.lower() in ('true', '1', 'yes')
        except ValueError:
            print(f"Warning: Could not convert environment variable {env_var_name} to expected type. Using default.")
        return env_var

    return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Allows accessing configuration values using dictionary-like syntax (e.g., config['key']).
        Raises a KeyError if not found.
        """
        if key not in self.config:
            raise KeyError(f"Configuration key '{key}' not found in {self.config_path}")
        return self.config[key]


    def __contains__(self, key:str) -> bool:
        """
        Checks if key is in the config.
        """
        return key in self.config


# Example usage (you would typically do this in your main training script):
# config = Config('config.yaml')
# learning_rate = config.get('learning_rate', 0.001)  # Get with default value
# batch_size = config['batch_size']  # Get using dictionary-like access
# if 'dataset_path' in config: #check if exists.
#    dataset_path = config['dataset_path']