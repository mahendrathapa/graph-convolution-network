import yaml
from src.constant import PROJECT_DIR


class TaskConfig:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def load_config(cls):

        config_path = PROJECT_DIR / "config/task_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        config = {k: v.get("value") for k, v in config.items()}

        return TaskConfig(**config)

    def __str__(self):
        return str(self.__dict__)
