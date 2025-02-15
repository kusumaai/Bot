#! /usr/bin/env python3
# tests/unit/test_rate_limiter.py
"""
Module: tests.unit
Provides unit testing functionality for the rate limiter module.
"""
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from exchanges.rate_limiter import RateLimitConfig, RateLimiter
from utils.error_handler import RateLimitExceeded


@pytest.mark.asyncio
async def test_rate_limiter_allow_requests():
    """Test that RateLimiter allows requests within the rate limit."""
    config = RateLimitConfig(max_requests=3, time_window=1)  # 3 requests per second
    limiter = RateLimiter(config)

    for _ in range(3):
        assert await limiter.acquire() is True
    # Fourth request should raise RateLimitExceeded
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire()


@pytest.mark.asyncio
async def test_rate_limiter_reset_after_window():
    """Test that RateLimiter resets after the time window."""
    config = RateLimitConfig(max_requests=2, time_window=1)  # 2 requests per second
    limiter = RateLimiter(config)

    for _ in range(2):
        assert await limiter.acquire() is True
    # Third request should raise RateLimitExceeded
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire()

    # Wait for window to reset
    await asyncio.sleep(1.1)

    # Requests should be allowed again
    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire()
