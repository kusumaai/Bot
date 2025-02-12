import asyncio
import time
from typing import Dict
import logging

from trading.exceptions import RateLimitExceeded

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
    def __init__(self, limits: Dict[str, RateLimit]):
        self.limits = limits

    async def acquire(self, key: str):
        if key in self.limits:
            await self.limits[key].acquire()
        else:
            raise ValueError(f"Rate limit key '{key}' not defined.") 