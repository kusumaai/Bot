#!/usr/bin/env python3
"""
PyTest configuration and shared fixtures
"""
import pytest
import asyncio
import logging
import os
import json
from decimal import Decimal
from typing import Any, Dict, AsyncGenerator, List
from pathlib import Path
import aiosqlite
import yaml
from datetime import datetime, timedelta

from database.database import DBConnection
from utils.error_handler import handle_error
from config.risk_config import RiskConfig
from database.connection import DatabaseConnection
from database.queries import DatabaseQueries
from utils.logger import setup_logging
from utils.error_handler import ErrorHandler

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration"""
    return {
        'database': {
            'path': 'data/test.db',
            'pool_size': 3,
            'max_connections': 5
        },
        'logging': {
            'level': 'DEBUG',
            'file_path': None
        },
        'timeframe': '15m',
        'risk_factor': Decimal('0.01'),
        'emergency_stop_pct': Decimal('-3.0'),
        'max_position_size': Decimal('0.1'),
        'max_positions': 3
    }

@pytest.fixture(scope="session")
async def test_context(test_config) -> Any:
    """Initialize test context with database and logger"""
    class TestContext:
        def __init__(self):
            # Set up logging
            self.logger = setup_logging(name="TestLogger", level=test_config['logging']['level'])
            
            # Database settings
            self.db_path = test_config['database']['path']
            self.db_connection = DatabaseConnection(
                db_path=self.db_path,
                logger=self.logger,
                pool_size=test_config['database']['pool_size'],
                max_connections=test_config['database']['max_connections']
            )
            self.db_queries = DatabaseQueries(self.db_connection, self.logger)
        
        async def setup(self):
            """Set up database with required tables"""
            await self.db_queries.initialize_tables()
        
        async def teardown(self):
            """Close database connections and clean up"""
            await self.db_connection.close_all()
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
    
    ctx = TestContext()
    await ctx.setup()
    yield ctx
    await ctx.teardown()

@pytest.fixture
async def db_connection(test_context) -> DatabaseConnection:
    """Provide a database connection fixture"""
    return test_context.db_connection

@pytest.fixture
async def db_queries(test_context) -> DatabaseQueries:
    """Provide a DatabaseQueries instance"""
    return test_context.db_queries

@pytest.fixture
def logger(test_context) -> logging.Logger:
    """Provide a logger fixture"""
    return test_context.logger

@pytest.fixture
def sample_candles() -> List[Dict[str, Any]]:
    """Provide sample market candle data"""
    return [
        {
            'timestamp': int((datetime.utcnow() - timedelta(minutes=i)).timestamp()),
            'open': 35000.0 + i,
            'high': 35100.0 + i,
            'low': 34900.0 + i,
            'close': 35050.0 + i,
            'volume': 10.0
        } for i in range(20)
    ]

@pytest.fixture
def sample_trades() -> List[Dict[str, Any]]:
    """Provide sample trade data"""
    return [
        {
            'id': 'trade001',
            'symbol': 'BTC/USDT',
            'entry_price': '50000',
            'size': '1.0',
            'side': 'buy',
            'strategy': 'StrategyA',
            'metadata': {'note': 'Initial buy'}
        },
        {
            'id': 'trade002',
            'symbol': 'ETH/USDT',
            'entry_price': '3000',
            'size': '10.0',
            'side': 'sell',
            'strategy': 'StrategyB',
            'metadata': {'note': 'Short position'}
        }
    ]

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Provide comprehensive test data"""
    return {
        "candles": [
            {
                "symbol": "BTC/USDT",
                "timeframe": "15m",
                "timestamp": 1600000000000,
                "open": 10000.0,
                "high": 10100.0,
                "low": 9900.0,
                "close": 10050.0,
                "volume": 100.0
            }
        ],
        "trades": [
            {
                "id": "trade123",
                "symbol": "BTC/USDT",
                "direction": "long",
                "entry_price": 10000.0,
                "exit_price": 10500.0,
                "size": 1.0,
                "pnl": 500.0,
                "entry_time": 1600000000,
                "exit_time": 1600001000,
                "status": "closed"
            }
        ]
    }

@pytest.fixture
def exchange_credentials():
    """Get exchange credentials from environment variables for testing"""
    return {
        'api_key': os.getenv('TEST_EXCHANGE_API_KEY', 'test_key'),
        'api_secret': os.getenv('TEST_EXCHANGE_SECRET', 'test_secret'),
        'exchange_id': os.getenv('TEST_EXCHANGE_ID', 'binance')
    }

@pytest.fixture
def risk_config():
    config = {
        "max_leverage": "2.0",
        "max_drawdown": "0.1",
        "max_daily_loss": "0.03",
        "ratchet_thresholds": ["1.0", "2.0"],
        "ratchet_lock_ins": ["0.5", "1.0"],
        "emergency_stop_pct": "-2",
        "trailing_stop_pct": "1.5",
        "max_hold_hours": "8",
        "max_position_pct": "10",
        "initial_balance": "10000"
    }
    return RiskConfig.from_config(config)