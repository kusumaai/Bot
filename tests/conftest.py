#!/usr/bin/env python3
"""
Module: tests/conftest.py
Test fixtures and configuration
"""

import os
import sys

# Determine the project root directory and the src directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')

# Add the src directory to sys.path so that modules inside src can be imported as top-level packages
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Optionally, also add the project root if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import asyncio
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# It is assumed that each directory in src (database, execution, risk, trading, utils)
# contains an __init__.py file to mark it as a package.
from database.database import DatabaseConnection
from execution.exchange_interface import ExchangeInterface
from execution.market_data import MarketData
from risk.manager import RiskManager
from trading.portfolio import PortfolioManager
from trading.circuit_breaker import CircuitBreaker
from trading.ratchet import RatchetManager
from utils.health_monitor import HealthMonitor
from utils.logger import setup_logging
from utils.numeric_handler import NumericHandler
from utils.error_handler import ExchangeError, RiskError, DatabaseError

# Example fixture to provide a mock trading context to tests
class MockContext:
    """Mock trading context for testing"""
    
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
            "database": {"path": "data/trading.db"}
        }
        self.db_pool = "data/candles.db"

@pytest.fixture
async def mock_db():
    """Mock database connection"""
    db = AsyncMock(spec=DatabaseConnection)
    db.pool = AsyncMock()
    db.pool.acquire = AsyncMock()
    
    # Setup mock connection and cursor
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.fetchall.return_value = [(1, 'test')]
    mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
    mock_conn.__aenter__.return_value = mock_conn
    db.pool.acquire.return_value = mock_conn
    
    # Add common database methods
    db.execute = AsyncMock(return_value=True)
    db.fetch_one = AsyncMock(return_value={'id': 1})
    db.fetch_all = AsyncMock(return_value=[{'id': 1}])
    
    return db

@pytest.fixture
async def mock_exchange():
    """Mock exchange interface"""
    exchange = AsyncMock()
    exchange.fetch_ticker = AsyncMock(return_value={'last': Decimal('50000')})
    exchange.create_order = AsyncMock(return_value={'id': 'test_order', 'status': 'open'})
    exchange.fetch_order = AsyncMock(return_value={'status': 'closed'})
    exchange.fetch_balance = AsyncMock(return_value={'free': {'USDT': Decimal('10000')}})
    exchange.ping = AsyncMock(return_value=None)
    exchange.has = {'fetchTicker': True, 'createOrder': True, 'fetchBalance': True}
    return exchange

@pytest.fixture
async def risk_manager(trading_context):
    """Create mock risk manager"""
    rm = RiskManager(trading_context)
    await rm.initialize()
    return rm

@pytest.fixture
async def portfolio_manager(trading_context, risk_manager):
    """Create mock portfolio manager"""
    pm = PortfolioManager(trading_context)
    trading_context.risk_manager = risk_manager
    await pm.initialize()
    return pm

@pytest.fixture
async def exchange_interface(trading_context, mock_exchange):
    """Create mock exchange interface"""
    ei = ExchangeInterface(trading_context)
    ei.exchange = mock_exchange
    await ei.initialize()
    return ei

@pytest.fixture
async def market_data(trading_context, exchange_interface):
    """Create mock market data service"""
    md = MarketData(trading_context)
    trading_context.exchange_interface = exchange_interface
    await md.initialize()
    return md

@pytest.fixture
async def health_monitor(trading_context):
    """Create mock health monitor"""
    hm = HealthMonitor(trading_context)
    await hm.initialize()
    return hm

@pytest.fixture
async def circuit_breaker(trading_context):
    """Create mock circuit breaker"""
    cb = CircuitBreaker(trading_context)
    await cb.initialize()
    return cb

@pytest.fixture
async def ratchet_manager(trading_context):
    """Create mock ratchet manager"""
    rm = RatchetManager(trading_context)
    await rm.initialize()
    return rm

@pytest.fixture
def trading_context(mock_db):
    """Create mock trading context with components"""
    ctx = MockContext()
    ctx.db_connection = mock_db
    return ctx

@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()