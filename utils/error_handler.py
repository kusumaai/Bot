#!/usr/bin/env python3
"""
Module: utils/error_handler.py
Centralized error handling utility
"""

import logging
import traceback
from typing import Optional
from datetime import datetime

def handle_error(
    error: Exception,
    location: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Centralized error handling function that logs errors with context
    
    Args:
        error: The exception that was caught
        location: String indicating where the error occurred
        logger: Optional logger instance. If None, uses root logger
    """
    if logger is None:
        logger = logging.getLogger()
        
    timestamp = datetime.now().isoformat()
    error_msg = f"Error in {location}: {str(error)}"
    
    # Log the basic error
    logger.error(error_msg)
    
    # Log the full traceback at debug level
    logger.debug(f"Traceback for error in {location}:\n{''.join(traceback.format_tb(error.__traceback__))}")
    
    # Log additional context if available
    if hasattr(error, 'response'):
        logger.debug(f"Error response data: {error.response}") 