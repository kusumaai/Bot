#!/usr/bin/env python3
"""
Module: tests/conftest.py
Test fixtures and configuration
"""
import os
import sys
import pytest
import asyncio
import logging

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any
from database.connection import DatabaseConnection
from execution.exchange_interface import ExchangeInterface
from execution.market_data import MarketData
from risk.manager import RiskManager
from trading.portfolio import PortfolioManager
from trading.circuit_breaker import CircuitBreaker
from trading.ratchet import RatchetManager
from utils.health_monitor import HealthMonitor
from utils.logger import setup_logging
from utils.numeric import NumericHandler
from utils.error_handler import ExchangeError, RiskError, DatabaseError

class MockContext:
    """Mock trading context for testing"""
    def __init__(self):
        self.logger = setup_logging('test')
        self.config = {
            'database': {
                'path': 'data/test_trading.db'  # Use SQLite for tests
            },
            'exchange': {
                'name': 'binance',
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'paper_mode': True,
                'rate_limit_per_second': 5
            },
            'risk_limits': {
                'max_position_size': '1000',
                'max_drawdown': '0.1',
                'emergency_stop_pct': '0.15',
                'max_leverage': '2.0',
                'max_daily_loss': '0.03'
            },
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'paper_mode': True,
            'initial_balance': '10000'
        }
        self.running = True
        self.initialized = False
        self.nh = NumericHandler()

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

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()