#! /usr/bin/env python3
# src/exchanges/rate_limiter.py
"""
Module: src.exchanges
Provides rate limiting implementation.
"""
import asyncio
import logging
import time
from typing import Dict

from src.utils.exceptions import RateLimitExceeded

logger = logging.getLogger(__name__)


class RateLimit:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = 0
        self.start_time = time.time()

    async def acquire(self):
        current_time = time.time()
        if current_time - self.start_time > self.period:
            self.calls = 0
            self.start_time = current_time
        if self.calls >= self.max_calls:
            logger.warning("Rate limit exceeded.")
            raise RateLimitExceeded("Rate limit exceeded.")
        self.calls += 1
        return True


class RateLimiter:
    def __init__(self, rate_limits):
        # Allow initialization with a RateLimitConfig or a dictionary of RateLimit objects
        from .rate_limiter import RateLimitConfig

        if isinstance(rate_limits, RateLimitConfig):
            # Create a default rate limit with key 'default'
            self.rate_limits = {
                "default": RateLimit(
                    rate_limits.max_requests, int(rate_limits.time_window)
                )
            }
        else:
            self.rate_limits = rate_limits
        self.logger = logging.getLogger(__name__)

    async def limit(self, key: str):
        if key in self.rate_limits:
            await self.rate_limits[key].acquire()

    async def acquire(self, key: str = "default"):
        if key in self.rate_limits:
            return await self.rate_limits[key].acquire()
        else:
            raise ValueError(f"Rate limit key '{key}' not defined.")


class RateLimitConfig:
    def __init__(self, max_requests: int, time_window: float):
        """
        Initialize rate limit configuration

        Args:
            max_requests (int): Maximum number of requests allowed in the time window
            time_window (float): Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
