#! /usr/bin/env python3
# tests/conftest.py
"""
Module: tests
Provides test configuration and shared fixtures for all test types.
"""
import asyncio
import logging
import os
import sys
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exchanges.exchange_manager import ExchangeManager
from execution.exchange_interface import ExchangeInterface
from execution.market_data import MarketData
from risk.limits import RiskLimits
from risk.manager import RiskManager
from trading.circuit_breaker import CircuitBreaker
from trading.portfolio import PortfolioManager
from trading.ratchet import RatchetManager
from utils.error_handler import DatabaseError, ExchangeError, RiskError
from utils.health_monitor import HealthMonitor
from utils.logger import setup_logging
from utils.numeric_handler import NumericHandler

# Import database fixtures from unit tests
from .unit.conftest import (
    base_db_connection,
    base_db_queries,
    mock_db_error_connection,
    mock_db_error_queries,
    sample_candle_data,
    sample_position_data,
    sample_trade_data,
    sample_trade_signal_data,
)


class MockContext:
    """Mock trading context for testing."""

    def __init__(self):
        self.logger = logging.getLogger("Test")
        self.config = {
            "timeframe": "15m",
            "emergency_stop_pct": -3,
            "ratchet_thresholds": [2, 4, 6],
            "ratchet_lock_ins": [1, 2, 3],
            "kelly_scaling": 0.5,
            "initial_balance": 10000,
            "risk_factor": 0.1,
            "database": {"path": "data/trading.db"},
        }
        self.db_pool = "data/candles.db"


@pytest.fixture
def mock_context():
    """Create base mock context with required components."""
    ctx = MagicMock()
    ctx.logger = MagicMock()
    ctx.db_connection = base_db_connection
    ctx.exchange_interface = MagicMock()
    ctx.market_data = MagicMock()
    ctx.config = {
        "exchange_id": "binance",
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "paper_mode": True,
        "database": {"path": "data/test_trading.db"},
        "initial_balance": "10000",
        "rate_limit_per_second": 5,
    }
    return ctx


@pytest.fixture
async def mock_exchange():
    """Mock exchange interface."""
    exchange = AsyncMock()
    exchange.fetch_ticker = AsyncMock(return_value={"last": Decimal("50000")})
    exchange.create_order = AsyncMock(
        return_value={"id": "test_order", "status": "open"}
    )
    exchange.fetch_order = AsyncMock(return_value={"status": "closed"})
    exchange.fetch_balance = AsyncMock(
        return_value={"free": {"USDT": Decimal("10000")}}
    )
    exchange.ping = AsyncMock(return_value=None)
    exchange.has = {"fetchTicker": True, "createOrder": True, "fetchBalance": True}
    return exchange


@pytest.fixture
async def risk_manager(trading_context):
    """Create mock risk manager."""
    rm = RiskManager(trading_context)
    await rm.initialize()
    return rm


@pytest.fixture
async def portfolio_manager(trading_context, risk_manager):
    """Create mock portfolio manager."""
    pm = PortfolioManager(trading_context)
    trading_context.risk_manager = risk_manager
    await pm.initialize()
    return pm


@pytest.fixture
async def exchange_interface(trading_context, mock_exchange):
    """Create mock exchange interface."""
    ei = ExchangeInterface(trading_context)
    ei.exchange = mock_exchange
    await ei.initialize()
    return ei


@pytest.fixture
async def market_data(trading_context, exchange_interface):
    """Create mock market data service."""
    md = MarketData(trading_context)
    trading_context.exchange_interface = exchange_interface
    await md.initialize()
    return md


@pytest.fixture
async def health_monitor(mock_context):
    """Create and initialize health monitor with mock context."""
    hm = HealthMonitor(mock_context)
    await hm.initialize()
    return hm


@pytest.fixture
async def circuit_breaker(trading_context):
    """Create mock circuit breaker."""
    cb = CircuitBreaker(trading_context)
    await cb.initialize()
    return cb


@pytest.fixture
async def ratchet_manager(trading_context):
    """Create mock ratchet manager."""
    rm = RatchetManager(trading_context)
    await rm.initialize()
    return rm


@pytest.fixture
def trading_context(base_db_connection):
    """Create mock trading context with components."""
    ctx = MockContext()
    ctx.db_connection = base_db_connection
    return ctx


@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def risk_limits():
    """Create risk limits instance."""
    return RiskLimits(
        max_value=Decimal("1000"),
        max_correlation=Decimal("0.75"),
        min_liquidity=Decimal("10000"),
        max_position_size=Decimal("500"),
        min_position_size=Decimal("10"),
        risk_factor=Decimal("0.01"),
    )


@pytest.fixture
async def mock_exchange_interface():
    """Standardized mock exchange interface."""
    interface = MagicMock(spec=ExchangeInterface)
    interface.exchange = MagicMock()
    interface.exchange.ping = AsyncMock(return_value=True)
    interface.execute_trade = AsyncMock(
        return_value={"success": True, "order_id": "test_order"}
    )
    interface.create_order = AsyncMock(
        return_value={
            "id": "order_test",
            "symbol": "BTC/USDT",
            "status": "open",
            "price": "50000",
            "amount": "0.1",
        }
    )
    interface.fetch_market_data = AsyncMock(
        return_value={"BTC/USDT": {"price": Decimal("50000")}}
    )
    interface.cancel_trade = AsyncMock()
    interface.get_order_status = AsyncMock()
    return interface


@pytest.fixture
async def mock_exchange_manager():
    """Standardized mock exchange manager."""
    manager = MagicMock(spec=ExchangeManager)
    manager.exchange = AsyncMock()
    manager.exchange.ping = AsyncMock(return_value=True)
    manager.exchange.create_order = AsyncMock(
        return_value={
            "id": "order_test",
            "symbol": "BTC/USDT",
            "status": "open",
            "price": "50000",
            "amount": "0.1",
        }
    )
    return manager
