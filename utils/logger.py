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
    
    # Color codes
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
        
        self.FORMATS = {
            logging.DEBUG: self.grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + self.reset,
            logging.INFO: self.blue + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + self.reset,
            logging.WARNING: self.yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + self.reset,
            logging.ERROR: self.red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + self.reset,
            logging.CRITICAL: self.bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + self.reset
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional color"""
        try:
            # Add extra fields if present
            if hasattr(record, 'extra_data'):
                record.msg = f"{record.msg} - {json.dumps(record.extra_data)}"
                
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            
            # Add stack trace for errors
            if record.levelno >= logging.ERROR:
                if record.exc_info:
                    record.msg = f"{record.msg}\n{traceback.format_exception(*record.exc_info)}"
                    
            return formatter.format(record)
            
        except Exception as e:
            handle_error(e, "CustomFormatter.format")
            return str(record.msg)

def setup_logger(
    level: str = "INFO",
    output: str = "stdout",
    ctx: Optional[Any] = None,
    name: str = "TradingBot",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up and return a logger instance with the specified configuration.
    
    Args:
        level (str): Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        output (str): Output destination ("stdout" or file path)
        ctx (Any): Optional context object with config overrides
        name (str): Logger name
        max_bytes (int): Maximum log file size before rotation
        backup_count (int): Number of backup files to keep
        
    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        # Get config overrides if ctx provided
        if ctx and hasattr(ctx, "config"):
            log_cfg = ctx.config.get("log_settings", {})
            level = log_cfg.get("level", level)
            output = log_cfg.get("output", output)
            max_bytes = log_cfg.get("max_bytes", max_bytes)
            backup_count = log_cfg.get("backup_count", backup_count)

        # Create logger
        logger = logging.getLogger(name)

        # Set level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create handler
        if output.lower() == "stdout":
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(CustomFormatter())
        else:
            # Ensure log directory exists
            log_dir = os.path.dirname(output)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            # Set up rotating file handler
            handler = logging.handlers.RotatingFileHandler(
                output,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        logger.addHandler(handler)
        
        # Add error file handler for ERROR and CRITICAL
        if output.lower() != "stdout":
            error_file = str(Path(output).parent / "error.log")
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s'
            ))
            logger.addHandler(error_handler)

        return logger

    except Exception as e:
        handle_error(e, "setup_logger")
        # Fallback to basic console logger
        basic_logger = logging.getLogger(name)
        basic_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        basic_logger.addHandler(handler)
        return basic_logger

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
