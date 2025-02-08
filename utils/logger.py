#!/usr/bin/env python3
"""
Module: utils/logger.py
"""

import logging
import sys
import os


def setup_logger(level: str = "INFO", output: str = "stdout", ctx=None) -> logging.Logger:
    """
    Set up and return a logger instance with the specified log level and output destination.
    Optionally, read overrides from ctx.config["log_settings"] if ctx is provided.

    Args:
        level (str): Log level as a string (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        output (str): Output destination. Use "stdout" for console output, or provide a file path for file logging.
        ctx (object, optional): An optional context object with config. If present, can override 'level' or 'output'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # If ctx is provided, look for config overrides
    if ctx and hasattr(ctx, "config"):
        log_cfg = ctx.config.get("log_settings", {})
        level = log_cfg.get("level", level)
        output = log_cfg.get("output", output)

    # Create a logger with the designated name.
    logger = logging.getLogger("TradingBot")

    # Set the log level; default to INFO if the provided level is invalid.
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Clear any existing handlers to avoid duplicate logging.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define the log message format.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create the appropriate handler based on the output parameter.
    if output.lower() == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        # Ensure the directory for the log file exists.
        log_dir = os.path.dirname(output)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(output)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
