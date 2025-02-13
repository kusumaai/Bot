#!/usr/bin/env python3
"""
Module: utils/logger.py
Logging configuration and setup
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from typing import Optional

def setup_logging(name: str, level: str, log_file: Optional[str] = "logs/KillaBot.log"):
    """Set up logging with specified level and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Stream Handler (e.g., Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File Handler (if log_file is specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    if name:
        return logging.getLogger(name)
    return logging.getLogger()
