import logging
import os
import sys
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
logger_path = os.path.abspath(os.path.join(base_folder, "./Log"))
if not os.path.exists(logger_path):
    os.mkdir(logger_path)


def getLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"[{name}] logger in ResNet")
    logger.setLevel(logging.INFO)
    terminal_handler = logging.StreamHandler(sys.stderr)
    terminal_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        os.path.abspath(os.path.join(logger_path, f"./{name}.log")),
        mode="w",
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)

    return logger
