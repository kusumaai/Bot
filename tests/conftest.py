#!/usr/bin/env python3
"""
PyTest configuration and shared fixtures
"""
import pytest
import logging
import os
import json
from decimal import Decimal
from typing import Any, Dict
from pathlib import Path

from database.database import DBConnection
from utils.error_handler import handle_error

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