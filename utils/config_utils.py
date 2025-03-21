import yaml
from typing import Dict, Any, Optional

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
        """Loads the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_path}")
            return {}  # Return an empty dictionary
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a configuration value by key.

        Args:
            key: The configuration key.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value, or the default value if the key is not found.
        """
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