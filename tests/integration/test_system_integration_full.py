#! /usr/bin/env python3
#tests/integration/test_system_integration_full.py
"""
Module: tests.integration
Provides integration testing functionality for the full system.
Maybe needs to merge with test_main_system.py
"""
import pytest
import asyncio
from src.startup.system_init import SystemInitializer
from src.database.connection import DatabaseConnection
from src.exchanges.exchange_manager import ExchangeManager
from src.utils.logger import setup_logging

class DummyDBConnection:
    async def verify_connection(self):
        return True

class DummyExchangeManager:
    async def verify_exchange_connection(self):
        return True

@pytest.fixture
def dummy_db_connection():
    return DummyDBConnection()

@pytest.fixture
def dummy_exchange_manager():
    return DummyExchangeManager()

@pytest.fixture
def logger():
    return setup_logging("systemIntegration", "DEBUG")

@pytest.fixture
def system_initializer(dummy_db_connection, dummy_exchange_manager, logger):
    return SystemInitializer(dummy_db_connection, dummy_exchange_manager, logger)

@pytest.mark.asyncio
async def test_system_initialization_full(system_initializer):
    result = await system_initializer.initialize_system()
    assert result is True 