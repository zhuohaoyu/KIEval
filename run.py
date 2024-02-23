import logging
import argparse
import json
import os
import socket
import sys
import re

from datetime import datetime
from colorlog import ColoredFormatter


class ColorFilter(logging.Filter):
    def filter(self, record):
        # Define a regex pattern to match ANSI color escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        # Remove ANSI color escape sequences from record.msg
        record.msg = ansi_escape.sub("", record.msg)
        return True


class SuffixFilter(logging.Filter):
    def filter(self, record):
        if "." in record.name:
            record.name_last = record.name.rsplit(".", 1)[-1]
        else:
            record.name_last = record.name
        return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file")
    parser.add_argument("-l", "--log", help="path to log file", default="./logs")

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()

    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s %(bold)s%(levelname)s%(reset)s %(log_color)s%(thin)s%(name_last)-16s %(reset)s %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    os.makedirs(args.log, exist_ok=True)
    log_filename = os.path.join(
        args.log, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    )

    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name_last)-16s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(SuffixFilter())
    file_handler.addFilter(ColorFilter())
    logger.addHandler(file_handler)

    logger.info(f"Loading config from {args.config}")

    config = json.load(open(args.config, "r"))

    if "run_name" not in config:
        config["run_name"] = os.path.split(args.config)[-1].replace(".", "_")

    # log hostname, user, python environment variables, ... to log file
    logger.info(f"Config: \n{json.dumps(config, indent=2, ensure_ascii=False)}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Environment Variables: {os.environ}")
    logger.info(f"Command: {sys.argv}")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler.addFilter(SuffixFilter())

    logger.info(f'Running SimpleEvaluationPipeline, run_name: {config["run_name"]}')

    from kieval.core.pipelines import EvaluationPipeline

    pipeline = EvaluationPipeline(config)
    pipeline.run()
