import asyncio
import time
from typing import Dict
import logging

from utils.exceptions import RateLimitExceeded

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

class RateLimiter:
    def __init__(self, rate_limits: Dict[str, RateLimit]):
        self.rate_limits = rate_limits
        self.logger = logging.getLogger(__name__)

    async def limit(self, key: str):
        if key in self.rate_limits:
            await self.rate_limits[key].acquire()

    async def acquire(self, key: str):
        if key in self.rate_limits:
            await self.rate_limits[key].acquire()
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