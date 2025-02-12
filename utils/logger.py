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

def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Set up logging with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # Prevent double logging

    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        log_output = os.getenv("LOG_OUTPUT", "logs/trading_bot.log")
        os.makedirs(os.path.dirname(log_output), exist_ok=True)
        fh = logging.FileHandler(log_output)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

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
