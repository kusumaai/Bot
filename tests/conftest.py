# tests/conftest.py
"""
PyTest configuration and shared fixtures
"""
import pytest
import logging
from typing import Any

@pytest.fixture
def test_context() -> Any:
    """Provide a test context with minimal configuration"""
    class TestContext:
        def __init__(self):
            self.logger = logging.getLogger("Test")
            self.config = {
                "timeframe": "15m",
                "emergency_stop_pct": -3,
                "ratchet_thresholds": [2, 4, 6],
                "ratchet_lock_ins": [1, 2, 3],
                "kelly_scaling": 0.5,
                "initial_balance": 10000,
                "risk_factor": 0.1
            }
            self.db_pool = "data/candles.db"
    
    return TestContext()