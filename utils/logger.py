#!/usr/bin/env python3
"""
Module: utils/logger.py
Production logging configuration with rotation and formatting
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional, Any, Dict
from pathlib import Path
import json
from datetime import datetime
import traceback
from utils.error_handler import handle_error

class CustomFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            record.color = self.COLORS.get(record.levelname, '')
            record.reset = self.RESET
        else:
            record.color = record.reset = ''
            
        return super().format(record)

def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    max_size: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure application logging with rotation and proper formatting
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        max_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        format_string: Optional custom format string
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format string
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_formatter = CustomFormatter(
        '%(color)s' + format_string + '%(reset)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Additional error file handler
        error_file = log_file.parent / f"error_{log_file.name}"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    return logger

def log_trade(logger: logging.Logger, trade_data: Dict[str, Any]) -> None:
    """
    Log trade information with proper formatting
    
    Args:
        logger (logging.Logger): Logger instance
        trade_data (Dict[str, Any]): Trade data to log
    """
    try:
        # Format trade data
        formatted_trade = {
            "trade_id": trade_data.get("trade_id", ""),
            "symbol": trade_data.get("symbol", ""),
            "direction": trade_data.get("direction", ""),
            "entry_time": datetime.fromtimestamp(
                trade_data.get("entry_time", 0)
            ).isoformat(),
            "entry_price": float(trade_data.get("entry_price", 0)),
            "position_size": float(trade_data.get("position_size", 0)),
            "status": trade_data.get("status", ""),
            "pnl": float(trade_data.get("pnl", 0)) if trade_data.get("pnl") else None
        }
        
        # Add exit information if available
        if trade_data.get("exit_time"):
            formatted_trade.update({
                "exit_time": datetime.fromtimestamp(trade_data["exit_time"]).isoformat(),
                "exit_price": float(trade_data["exit_price"]),
                "exit_reason": trade_data.get("exit_reason", "")
            })
            
        logger.info(
            f"Trade Update - {formatted_trade['status']}",
            extra={"extra_data": formatted_trade}
        )
        
    except Exception as e:
        handle_error(e, "log_trade", logger=logger)
