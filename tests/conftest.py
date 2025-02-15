#! /usr/bin/env python3
#tests/conftest.py
"""
Module: tests
Provides unit testing functionality for the tests module.
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

#each directory in src contains an __init__.py file to mark it as a package.
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
from risk.limits import RiskLimits

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
#mock database connection
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
#exchange interface
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
#risk manager
@pytest.fixture
async def risk_manager(trading_context):
    """Create mock risk manager"""
    rm = RiskManager(trading_context)
    await rm.initialize()
    return rm
#portfolio manager
@pytest.fixture
async def portfolio_manager(trading_context, risk_manager):
    """Create mock portfolio manager"""
    pm = PortfolioManager(trading_context)
    trading_context.risk_manager = risk_manager
    await pm.initialize()
    return pm
#exchange interface
@pytest.fixture
async def exchange_interface(trading_context, mock_exchange):
    """Create mock exchange interface"""
    ei = ExchangeInterface(trading_context)
    ei.exchange = mock_exchange
    await ei.initialize()
    return ei
#market data
@pytest.fixture
async def market_data(trading_context, exchange_interface):
    """Create mock market data service"""
    md = MarketData(trading_context)
    trading_context.exchange_interface = exchange_interface
    await md.initialize()
    return md
#health monitor
@pytest.fixture
async def health_monitor(trading_context):
    """Create mock health monitor"""
    hm = HealthMonitor(trading_context)
    await hm.initialize()
    return hm
#circuit breaker
@pytest.fixture
async def circuit_breaker(trading_context):
    """Create mock circuit breaker"""
    cb = CircuitBreaker(trading_context)
    await cb.initialize()
    return cb
#ratchet manager
@pytest.fixture
async def ratchet_manager(trading_context):
    """Create mock ratchet manager"""
    rm = RatchetManager(trading_context)
    await rm.initialize()
    return rm
#numeric handler    
@pytest.fixture
async def numeric_handler(trading_context):
    """Create mock numeric handler"""
    nh = NumericHandler(trading_context)
    await nh.initialize()
    return nh
#trading context
@pytest.fixture
def trading_context(mock_db):
    """Create mock trading context with components"""
    ctx = MockContext()
    ctx.db_connection = mock_db
    return ctx
#event loop
@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
#logger
@pytest.fixture
def mock_logger():
    """Create a mock logger"""
    return MagicMock()
#database queries
@pytest.fixture
def mock_db_queries():
    """Create a mock database queries"""
    db_queries = MagicMock()
    return db_queries
#exchange interface
@pytest.fixture
def mock_exchange_interface():
    """Create a mock exchange interface"""
    exchange_interface = MagicMock()
    return exchange_interface
#market data
@pytest.fixture
def mock_market_data():
    """Create a mock market data"""
    market_data = MagicMock()
    return market_data
#health monitor
@pytest.fixture
def mock_health_monitor():
    """Create a mock health monitor"""
    health_monitor = MagicMock()
    return health_monitor
#circuit breaker
@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker"""
    circuit_breaker = MagicMock()
    return circuit_breaker
#ratchet manager
@pytest.fixture
def mock_ratchet_manager():
    """Create a mock ratchet manager"""
    ratchet_manager = MagicMock()
    return ratchet_manager
#numeric handler
@pytest.fixture
def mock_numeric_handler():
    """Create a mock numeric handler"""
    numeric_handler = MagicMock()
    return numeric_handler
#market data validation
@pytest.fixture
def mock_market_data_validation():
    """Create a mock market data validation"""
    market_data_validation = MagicMock()
    return market_data_validation
#math handler
@pytest.fixture
def mock_math_handler():
    """Create a mock math handler"""
    math_handler = MagicMock()
    return math_handler
#risk limits
@pytest.fixture
def mock_risk_limits():
    """Create a mock risk limits"""
    risk_limits = MagicMock()
    return risk_limits
#risk limits validation
@pytest.fixture
def mock_risk_limits_validation():
    """Create a mock risk limits validation"""
    risk_limits_validation = MagicMock()
    return risk_limits_validation
#risk limits
@pytest.fixture
def risk_limits():
    return RiskLimits(
        max_value=Decimal('1000'),
        max_correlation=Decimal('0.75'),
        min_liquidity=Decimal('10000'),
        max_position_size=Decimal('500'),
        min_position_size=Decimal('10'),
        risk_factor=Decimal('0.01')
    )
#database queries
@pytest.fixture
def db_queries():
    from src.database.queries import DatabaseQueries
    return MagicMock(spec=DatabaseQueries)
















