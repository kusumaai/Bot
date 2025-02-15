import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logging(
    name: str, level: str = "INFO", log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up centralized logging with console and file handlers.

    :param name: Logger name
    :param level: Logging level as a string (e.g., "INFO")
    :param log_dir: Directory where log files will be stored
    :return: Configured logger instance
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler with TimedRotatingFileHandler (daily rotation, 7 backups)
    file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = TimedRotatingFileHandler(
        file_path, when="midnight", interval=1, backupCount=7
    )
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
