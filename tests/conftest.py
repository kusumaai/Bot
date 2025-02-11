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

from database.database import DBConnection
from utils.error_handler import handle_error
from config.settings import Settings
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
            'path': 'test.db',
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
def logger():
    """Provide test logger"""
    return setup_logging("test_logger", level="DEBUG")

@pytest.fixture(scope="session")
def error_handler(logger):
    """Provide error handler instance"""
    return ErrorHandler(logger)

@pytest.fixture(scope="session")
async def db_connection(test_config, logger) -> AsyncGenerator[DatabaseConnection, None]:
    """
    Provide test database connection with temporary database
    """
    db_path = Path(test_config['database']['path'])
    
    # Create test database
    connection = DatabaseConnection(
        db_path=db_path,
        logger=logger,
        pool_size=test_config['database']['pool_size'],
        max_connections=test_config['database']['max_connections']
    )
    
    # Initialize schema
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                timeframe TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );
            
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                signal_type TEXT,
                direction TEXT,
                timestamp INTEGER,
                metadata TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                size REAL,
                status TEXT,
                timestamp INTEGER,
                metadata TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                context TEXT,
                error_type TEXT,
                error_message TEXT,
                traceback TEXT,
                metadata TEXT
            );
        """)
    
    await connection.initialize()
    yield connection
    
    # Cleanup
    await connection._pool.join()
    db_path.unlink(missing_ok=True)

@pytest.fixture
async def db_queries(db_connection, logger) -> DatabaseQueries:
    """Provide database queries instance"""
    return DatabaseQueries(db_connection, logger)

@pytest.fixture
async def sample_candles() -> List[Dict[str, Any]]:
    """Provide sample candle data for testing"""
    return [
        {
            'timestamp': 1625097600,
            'open': 35000.0,
            'high': 35100.0,
            'low': 34900.0,
            'close': 35050.0,
            'volume': 100.0
        },
        {
            'timestamp': 1625097900,
            'open': 35050.0,
            'high': 35200.0,
            'low': 35000.0,
            'close': 35150.0,
            'volume': 150.0
        }
    ]

@pytest.fixture
def mock_settings(test_config) -> Settings:
    """Provide mock settings instance"""
    class MockSettings(Settings):
        def __init__(self, config):
            self._settings = config
            
        def _load_config(self):
            pass
            
        def _validate_settings(self):
            pass
    
    return MockSettings(test_config)

@pytest.fixture
def test_context() -> Any:
    """Provide a test context with minimal configuration"""
    class TestContext:
        def __init__(self):
            # Set up logging
            self.logger = logging.getLogger("Test")
            self.logger.setLevel(logging.DEBUG)
            
            # Load test configuration
            self.config = {
                # Time settings
                "timeframe": "15m",
                "lookback_periods": 100,
                
                # Risk settings
                "emergency_stop_pct": Decimal("-3"),
                "max_drawdown_pct": Decimal("20"),
                "max_daily_loss": Decimal("5"),
                "risk_factor": Decimal("0.1"),
                "kelly_scaling": Decimal("0.5"),
                
                # Position settings
                "max_position_size": Decimal("0.2"),
                "min_position_size": Decimal("0.01"),
                "max_positions": 5,
                "max_correlation": Decimal("0.7"),
                
                # Strategy settings
                "ratchet_thresholds": [2, 4, 6],
                "ratchet_lock_ins": [1, 2, 3],
                "initial_balance": Decimal("10000"),
                
                # GA settings
                "ga_settings": {
                    "population_size": 50,
                    "generations": 10,
                    "mutation_rate": Decimal("0.1"),
                    "crossover_rate": Decimal("0.8"),
                    "tournament_size": 3,
                    "min_fitness": Decimal("0.05")
                },
                
                # ML settings
                "ml_settings": {
                    "min_probability": Decimal("0.55"),
                    "prediction_threshold": Decimal("0.6"),
                    "confidence_threshold": Decimal("0.7")
                },
                
                # Exchange settings
                "exchange": "binance",
                "commission_rate": Decimal("0.001"),
                "market_list": ["BTC/USDT", "ETH/USDT"],
                
                # Database settings
                "database": {
                    "path": "data/test.db",
                    "backup_path": "data/backups"
                }
            }
            
            # Set up test database
            self.db_pool = self._initialize_test_db()
            
    def _initialize_test_db(self):
        """Initialize test database with required tables"""
        try:
            db_path = Path("data/test.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = DBConnection(str(db_path))
            
            # Create required tables
            tables = [
                """
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    pnl REAL,
                    entry_time INTEGER NOT NULL,
                    exit_time INTEGER,
                    status TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_performance (
                    id INTEGER PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL
                )
                """
            ]
            
            for table in tables:
                conn.execute(table)
                
            return str(db_path)
            
        except Exception as e:
            handle_error(e, "conftest._initialize_test_db")
            raise
            
    return TestContext()

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Provide sample test data"""
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

@pytest.fixture(autouse=True)
def cleanup_test_db(test_context):
    """Clean up test database after each test"""
    yield
    try:
        if os.path.exists(test_context.db_pool):
            os.remove(test_context.db_pool)
    except Exception as e:
        handle_error(e, "conftest.cleanup_test_db")