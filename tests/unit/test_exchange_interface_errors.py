#! /usr/bin/env python3
# tests/unit/test_exchange_interface_errors.py
"""
Module: tests.unit
Provides unit testing functionality for the exchange interface errors module.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from execution.exchange_interface import ExchangeInterface
from utils.error_handler import ExchangeError


@pytest.fixture
def fake_exchange_manager():
    fake = MagicMock()
    # Simulate a network failure when creating an order
    fake.exchange = MagicMock()
    fake.exchange.create_order = AsyncMock(side_effect=Exception("Network failure"))
    return fake


@pytest.fixture
def fake_risk_manager():
    return MagicMock()


@pytest.fixture
def fake_db_queries():
    return MagicMock()


@pytest.fixture
def fake_logger():
    import logging

    return logging.getLogger("test_exchange")


@pytest.fixture
def exchange_interface(
    fake_exchange_manager, fake_risk_manager, fake_db_queries, fake_logger
):
    return ExchangeInterface(
        fake_exchange_manager, fake_risk_manager, fake_db_queries, fake_logger
    )


@pytest.mark.asyncio
async def test_exchange_order_failure(exchange_interface):
    with pytest.raises(ExchangeError):
        await exchange_interface.execute_trade(
            symbol="BTC/USDT", side="buy", amount=1, order_type="limit", price=50000
        )
