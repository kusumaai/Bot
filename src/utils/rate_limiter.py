#! /usr/bin/env python3
#src/utils/rate_limiter.py
"""
Module: src.utils
Provides rate limiting functionality.
""" 
import asyncio
import logging
import time
import random
from typing import Dict, List

from exchanges.rate_limiter import RateLimit

class RateLimiter:
    def __init__(self, limits: Dict[str, RateLimit]):
        self.limits = limits
        self._locks: Dict[str, asyncio.Lock] = {
            endpoint: asyncio.Lock() for endpoint in limits.keys()
        }
        self.request_timestamps: Dict[str, List[float]] = {
            endpoint: [] for endpoint in limits.keys()
        }
        
    async def acquire(self, endpoint: str) -> None:
        """Thread-safe rate limit checking with backoff"""
        if endpoint not in self.limits:
            return
            
        try:
            async with self._locks[endpoint]:
                await self._check_rate_limit(endpoint)
        except Exception as e:
            logging.error(f"Rate limiter error: {e}")
            await asyncio.sleep(1)
            
    async def _check_rate_limit(self, endpoint: str) -> None:
        """Check and enforce rate limits with cleanup"""
        limit = self.limits[endpoint]
        now = time.time()
        window_start = now - limit.time_window
        
        # Clean old timestamps
        self.request_timestamps[endpoint] = [
            ts for ts in self.request_timestamps[endpoint]
            if ts > window_start
        ]
        
        # Implement exponential backoff if near limit
        while len(self.request_timestamps[endpoint]) >= limit.max_requests:
            sleep_time = self._calculate_backoff(endpoint)
            await asyncio.sleep(sleep_time)
            now = time.time()
            window_start = now - limit.time_window
            
        self.request_timestamps[endpoint].append(now)
        
    def _calculate_backoff(self, endpoint: str) -> float:
        """Calculate exponential backoff time"""
        attempt = len(self.request_timestamps[endpoint]) - self.limits[endpoint].max_requests
        return min(300, (2 ** attempt) + random.uniform(0, 1))  # Max 5 minutes 