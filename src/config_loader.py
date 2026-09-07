import yaml


def load_config(path="config/experiment_config.yaml"):
    """Load the experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)