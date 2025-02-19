#! /usr/bin/env python3
# src/exchanges/exchange_manager.py
"""
Module: exchanges/exchange_manager.py
Manages exchange connections and operations with proper rate limiting and error handling.
This version consolidates duplicate implementations and provides a consistent API.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

from src.database.database import DatabaseConnection, DatabaseQueries
from src.exchanges.actual_exchange import ActualExchange
from src.exchanges.paper_exchange import PaperExchange
from src.exchanges.rate_limiter import RateLimit, RateLimiter
from src.trading.exceptions import ExchangeAPIError, RateLimitExceeded
from src.utils.error_handler import ExchangeError, handle_error_async
from src.utils.numeric_handler import NumericHandler


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests_per_second: int
    max_requests_per_minute: int
    max_requests_per_hour: int
    max_requests_per_day: int
    backoff_factor: float = 1.5
    max_backoff: int = 300  # 5 minutes
    min_remaining_threshold: float = 0.1  # 10% remaining capacity threshold

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RateLimitConfig":
        """Create config from dictionary."""
        return cls(
            max_requests_per_second=int(config.get("max_requests_per_second", 10)),
            max_requests_per_minute=int(config.get("max_requests_per_minute", 600)),
            max_requests_per_hour=int(config.get("max_requests_per_hour", 36000)),
            max_requests_per_day=int(config.get("max_requests_per_day", 864000)),
            backoff_factor=float(config.get("backoff_factor", 1.5)),
            max_backoff=int(config.get("max_backoff", 300)),
            min_remaining_threshold=float(config.get("min_remaining_threshold", 0.1)),
        )


@dataclass
class RateLimit:
    """Rate limit configuration with enhanced tracking"""

    max_calls: int
    period: int  # seconds
    current_calls: int = 0
    last_reset: float = 0.0
    backoff_factor: float = 1.5
    max_backoff: int = 300  # 5 minutes
    min_remaining_threshold: float = 0.1  # 10% remaining capacity threshold


@dataclass
class RateLimitState:
    """Tracks the current state of rate limiting"""

    remaining_calls: int
    reset_time: float
    last_updated: float
    consecutive_429s: int = 0
    current_backoff: float = 1.0
    is_backoff_mode: bool = False


class RateLimiterContext:
    """Context manager for rate limiting."""

    def __init__(self, rate_limiter: "RateLimiter", category: str):
        self.rate_limiter = rate_limiter
        self.category = category

    async def __aenter__(self):
        await self.rate_limiter.acquire(self.category)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.rate_limiter.release(self.category)


class RateLimiter:
    """Enhanced rate limiter with proper backoff and pre-emptive checking."""

    def __init__(self, rate_limits: Dict[str, RateLimit]):
        self.rate_limits = rate_limits
        self._locks: Dict[str, asyncio.Lock] = {}
        self._states: Dict[str, RateLimitState] = {}
        self._call_times: Dict[str, List[float]] = {}
        self._last_429_time: Dict[str, float] = {}

        for category in rate_limits:
            self._locks[category] = asyncio.Lock()
            self._call_times[category] = []
            self._states[category] = RateLimitState(
                remaining_calls=rate_limits[category].max_calls,
                reset_time=time.time() + rate_limits[category].period,
                last_updated=time.time(),
            )

    def acquire_context(self, category: str) -> RateLimiterContext:
        """Get a context manager for rate limiting."""
        return RateLimiterContext(self, category)

    async def acquire(self, category: str):
        """Acquire a rate limit slot with enhanced error handling and backoff."""
        if category not in self.rate_limits:
            raise ValueError(f"Unknown rate limit category: {category}")

        limit = self.rate_limits[category]
        state = self._states[category]

        async with self._locks[category]:
            now = time.time()

            # Reset if period has elapsed
            if now >= state.reset_time:
                self._reset_state(category, now)

            # Check if we need to wait for reset
            if state.remaining_calls <= (
                limit.max_calls * limit.min_remaining_threshold
            ):
                wait_time = state.reset_time - now
                if wait_time > 0:
                    await self._handle_near_limit(category, wait_time)
                    return

            # Apply backoff if in backoff mode
            if state.is_backoff_mode:
                await self._apply_backoff(category, now)

            # Update state
            state.remaining_calls -= 1
            state.last_updated = now
            self._call_times[category].append(now)

    async def _handle_near_limit(self, category: str, base_wait_time: float):
        """Handle cases where we're near the rate limit."""
        state = self._states[category]
        limit = self.rate_limits[category]

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, 1)
        wait_time = base_wait_time * (1 + jitter * 0.1)  # 10% jitter

        # If we've hit 429s recently, increase wait time
        if state.consecutive_429s > 0:
            wait_time *= 1 + (state.consecutive_429s * 0.5)  # 50% increase per 429

        await asyncio.sleep(min(wait_time, limit.max_backoff))

    async def _apply_backoff(self, category: str, now: float):
        """Apply exponential backoff when needed."""
        state = self._states[category]
        limit = self.rate_limits[category]

        if state.is_backoff_mode:
            backoff_time = min(
                state.current_backoff * limit.backoff_factor, limit.max_backoff
            )
            state.current_backoff = backoff_time

            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(backoff_time * jitter)

    def _reset_state(self, category: str, now: float):
        """Reset rate limit state for a category."""
        limit = self.rate_limits[category]
        state = self._states[category]

        state.remaining_calls = limit.max_calls
        state.reset_time = now + limit.period
        state.last_updated = now

        # Clear old call times
        self._call_times[category] = [
            t for t in self._call_times[category] if now - t < limit.period
        ]

    async def handle_429(self, category: str):
        """Handle rate limit exceeded errors."""
        state = self._states[category]
        limit = self.rate_limits[category]

        async with self._locks[category]:
            state.consecutive_429s += 1
            state.is_backoff_mode = True
            state.current_backoff = max(
                limit.period / limit.max_calls,  # Base backoff
                state.current_backoff * limit.backoff_factor,
            )
            self._last_429_time[category] = time.time()

    def release(self, category: str):
        """Update state after successful call."""
        state = self._states[category]
        if state.consecutive_429s > 0:
            state.consecutive_429s = 0
            state.is_backoff_mode = False
            state.current_backoff = 1.0

    def get_state(self, category: str) -> Dict[str, Any]:
        """Get current rate limit state."""
        if category not in self._states:
            return {}

        state = self._states[category]
        limit = self.rate_limits[category]

        return {
            "remaining_calls": state.remaining_calls,
            "max_calls": limit.max_calls,
            "reset_in": state.reset_time - time.time(),
            "is_backoff_mode": state.is_backoff_mode,
            "consecutive_429s": state.consecutive_429s,
            "current_backoff": state.current_backoff,
        }


class ExchangeManager:
    """Manages exchange interactions with proper rate limiting and error handling"""

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = True,
        logger: Optional[logging.Logger] = None,
        db_queries: Optional[Any] = None,
    ):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.logger = logger or logging.getLogger(__name__)
        self.db_queries = db_queries
        self._markets = None
        self._last_market_update = None
        self._lock = asyncio.Lock()

        if sandbox:
            self.exchange = PaperExchange()
        else:
            # Handle live exchange initialization
            pass

        # Initialize Rate Limiter with proper context management
        self.rate_limiter = RateLimiter(
            rate_limits={
                "market": RateLimit(
                    max_calls=100,
                    period=60,
                    min_remaining_threshold=0.15,  # More conservative for market data
                ),
                "trade": RateLimit(
                    max_calls=50,
                    period=60,
                    min_remaining_threshold=0.2,  # More conservative for trades
                    backoff_factor=2.0,  # More aggressive backoff for trades
                ),
            }
        )

        self.numeric_handler = NumericHandler()

        if self.sandbox:
            # Set default paper mode balances and positions
            self.paper_balances = {"USDT": Decimal("10000.0")}
            self.paper_positions = {}

    async def initialize(self) -> bool:
        """Initialize exchange connection"""
        try:
            await self.exchange.load_markets()
            self.logger.info("Exchange initialized successfully")
            return True
        except ExchangeError as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during exchange initialization: {e}")
            raise ExchangeError(f"Exchange initialization failed: {str(e)}") from e

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Creates an order with enhanced rate limit handling."""
        async with self.rate_limiter.acquire_context("trade"):
            try:
                # Pre-emptively check rate limit state
                state = self.rate_limiter.get_state("trade")
                if state.get("is_backoff_mode"):
                    self.logger.warning(
                        f"Rate limit backoff in progress. Current backoff: {state['current_backoff']}s"
                    )

                order = await self.exchange.create_order(
                    symbol, order_type, side, amount, price
                )
                self.logger.info(f"Created order: {order}")
                return order

            except ccxt.RateLimitExceeded as e:
                await self.rate_limiter.handle_429("trade")
                raise ExchangeError(f"Rate limit exceeded: {str(e)}")
            except Exception as e:
                await handle_error_async(e, "ExchangeManager.create_order", self.logger)
                raise ExchangeError(f"Order creation failed: {str(e)}")

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker data with proper rate limiting."""
        async with self.rate_limiter.acquire_context("market"):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                if self.db_queries:
                    # Log ticker information to the database.
                    await self.db_queries.execute(
                        "INSERT INTO tickers (symbol, data) VALUES (?, ?)",
                        (symbol, str(ticker)),
                    )
                return ticker
            except Exception as e:
                raise ExchangeError(f"Failed to fetch ticker: {str(e)}") from e

    async def get_markets(self) -> Dict[str, Any]:
        """Fetch markets with caching and proper rate limiting."""
        async with self.rate_limiter.acquire_context("market"):
            try:
                if (
                    not self._markets
                    or not self._last_market_update
                    or (datetime.utcnow() - self._last_market_update).seconds > 3600
                ):
                    self._markets = await self.exchange.fetch_markets()
                    self._last_market_update = datetime.utcnow()
                return self._markets
            except Exception as e:
                await handle_error_async(e, "ExchangeManager.get_markets", self.logger)
                raise ExchangeError(f"Failed to fetch markets: {str(e)}") from e

    async def close(self):
        """Close exchange and database connections."""
        async with self._lock:
            if self.exchange:
                await self.exchange.close()
            if self.db_queries:
                await self.db_queries.close()
