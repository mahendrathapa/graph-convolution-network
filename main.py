import sys

import click
from loguru import logger

from src.config.task_config import TaskConfig
from src.task.cora_node_classification import CoraNodeClassification

logger.remove()
logger.add(sink=sys.stderr, level="INFO")


@click.group(name="main")
def main():
    pass


@main.command(name='cora-node-classification', help='Run coda node classification task')
def cora_node_classification():

    logger.info("Starting Task: Cora Node Classification")

    config = TaskConfig.load_config()
    model = CoraNodeClassification(config)
    model.run()

    logger.info("Completed Task: Cora Node Classification")


if __name__ == "__main__":
    main()
