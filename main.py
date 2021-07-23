import click

from src.config.task_config import TaskConfig
from src.task.cora_node_classification import CoraNodeClassification


@click.group(name="main")
def main():
    pass


@main.command(name='cora-node-classification', help='Run coda node classification task')
def cora_node_classification():
    config = TaskConfig.load_config()
    model = CoraNodeClassification(config)
    model.run()


if __name__ == "__main__":
    main()
