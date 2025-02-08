# tests/__init__.py
"""
Test package initialization.
Ensures the root directory is in the Python path.
"""
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# tests/unit/__init__.py
"""
Unit test package initialization
"""

# tests/integration/__init__.py
"""
Integration test package initialization
"""

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

# tests/README.md
"""
KillaBot Test Suite
==================

This directory contains the test suite for the KillaBot trading system.

Directory Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- validation_tests.py: Comprehensive validation suite
- conftest.py: PyTest configuration and shared fixtures

Running Tests:
-------------
From project root:
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/validation_tests.py

# Run with verbose output
pytest -v tests/
```

Writing Tests:
-------------
1. Unit tests go in tests/unit/
2. Integration tests go in tests/integration/
3. Use fixtures from conftest.py for shared resources
4. Follow existing test patterns for consistency
"""