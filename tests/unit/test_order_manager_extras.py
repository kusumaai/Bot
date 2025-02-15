#! /usr/bin/env python3
#tests/unit/test_order_manager_extras.py
"""
Module: tests.unit
Provides unit testing functionality for the order manager extras module.
""" 
import pytest
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from src.execution.order_manager import OrderManager
from src.utils.error_handler import OrderError

@pytest.fixture
def fake_exchange_interface():
    fake = MagicMock()
    fake.execute_trade = AsyncMock(return_value={'success': True, 'order_id': 'order123'})
    fake.cancel_trade = AsyncMock(return_value=True)
    return fake

@pytest.fixture
def fake_db_queries():
    fake = MagicMock()
    fake.store_order = AsyncMock(return_value=True)
    return fake

@pytest.fixture
def fake_logger():
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def order_manager(fake_exchange_interface, fake_db_queries, fake_logger):
    return OrderManager(fake_exchange_interface, fake_db_queries, fake_logger)

import asyncio

@pytest.mark.asyncio
async def test_successful_order(order_manager):
    result = await order_manager.place_order(
        symbol='BTC/USDT',
        side='sell',
        amount=Decimal('0.1'),
        order_type='market',
        price=Decimal('48000')
    )
    order_manager.db_queries.store_order.assert_awaited_once()
    assert result is True

@pytest.mark.asyncio
async def test_order_cancellation(order_manager):
    # Simulate execute_trade failure triggering cancellation and error propagation
    order_manager.exchange_interface.execute_trade.return_value = {'success': False, 'error': 'Order Error'}
    with pytest.raises(OrderError):
        await order_manager.place_order(
            symbol='BTC/USDT',
            side='sell',
            amount=Decimal('0.1'),
            order_type='limit',
            price=Decimal('48000')
        ) 