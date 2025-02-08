#!/usr/bin/env python3
"""
Module: utils/error_handler.py
"""

import logging
import traceback
import time
import asyncio
from functools import wraps
from typing import Callable, Any, TypeVar, Coroutine

T = TypeVar('T')


def handle_error(error: Exception, context: str, logger: logging.Logger) -> None:
    """
    Log an error with its context and stack trace.
    """
    error_message = f"Error in {context}: {error}"
    logger.error(error_message)
    logger.debug(traceback.format_exc())


def retry_wrapper(max_retries: int = 3, delay: int = 2) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a synchronous function upon failure.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            ctx = args[0] if args else None
            if hasattr(ctx, "config"):
                cfg = ctx.config.get("retry_settings", {})
                retries = cfg.get("max_retries", max_retries)
                base_delay = cfg.get("delay", delay)
            else:
                retries = max_retries
                base_delay = delay

            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    log = getattr(ctx, 'logger', logging.getLogger()) if ctx else logging.getLogger()
                    handle_error(exc, f"{func.__name__} (attempt {attempt+1})", log)
                    attempt += 1
                    if attempt < retries:
                        log.info(f"Retrying {func.__name__} in {base_delay * attempt} seconds...")
                        time.sleep(base_delay * attempt)
            raise Exception(f"Max retries reached in {func.__name__}")
        return wrapper
    return decorator


def async_retry_wrapper(max_retries: int = 3, delay: int = 2) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Decorator that retries an async function upon failure.
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            ctx = args[0] if args else None
            if hasattr(ctx, "config"):
                cfg = ctx.config.get("retry_settings", {})
                retries = cfg.get("max_retries", max_retries)
                base_delay = cfg.get("delay", delay)
            else:
                retries = max_retries
                base_delay = delay

            attempt = 0
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    log = getattr(ctx, 'logger', logging.getLogger()) if ctx else logging.getLogger()
                    handle_error(exc, f"{func.__name__} (attempt {attempt+1})", log)
                    attempt += 1
                    if attempt < retries:
                        log.info(f"Retrying async {func.__name__} in {base_delay * attempt} seconds...")
                        await asyncio.sleep(base_delay * attempt)
            raise Exception(f"Max async retries reached in {func.__name__}")
        return wrapper
    return decorator
