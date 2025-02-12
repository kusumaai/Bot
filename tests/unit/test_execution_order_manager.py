import logging
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from execution.order_manager import OrderManager
from execution.exchange_interface import ExchangeInterface
from utils.error_handler import ExchangeError, OrderError
from database.queries import DatabaseQueries


@pytest.fixture
def mock_exchange_interface():
    """Provide a mocked ExchangeInterface."""
    interface = MagicMock(spec=ExchangeInterface)
    interface.execute_trade = AsyncMock()
    interface.cancel_trade = AsyncMock()
    interface.get_order_status = AsyncMock()
    return interface


@pytest.fixture
def db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def order_manager(mock_exchange_interface, db_queries, logger):
    """Provide an OrderManager instance."""
    return OrderManager(
        exchange_interface=mock_exchange_interface,
        db_queries=db_queries,
        logger=logger
    )


@pytest.mark.asyncio
async def test_place_order_success(order_manager):
    """Test successful order placement."""
    order_details = {'id': 'order001', 'status': 'open'}
    order_manager.exchange_interface.execute_trade.return_value = {'success': True, 'order_id': 'order001'}
    order_manager.db_queries.store_order.return_value = True

    result = await order_manager.place_order(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.5'),
        order_type='limit',
        price=Decimal('50000')
    )
    assert result == True
    order_manager.exchange_interface.execute_trade.assert_awaited_once()
    order_manager.db_queries.store_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_place_order_risk_failure(order_manager):
    """Test order placement when risk validation fails."""
    order_manager.exchange_interface.execute_trade.return_value = {'success': False, 'error': 'Risk Validation Failed'}

    result = await order_manager.place_order(
        symbol='ETH/USDT',
        side='sell',
        amount=Decimal('2'),
        order_type='market',
        price=Decimal('3000')
    )
    assert result == False
    order_manager.exchange_interface.execute_trade.assert_awaited_once()
    order_manager.db_queries.store_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_place_order_exchange_error(order_manager):
    """Test order placement when exchange raises an error."""
    order_manager.exchange_interface.execute_trade.side_effect = ExchangeError("Exchange Error")

    result = await order_manager.place_order(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('1'),
        order_type='limit',
        price=Decimal('50000')
    )
    assert result == False
    order_manager.exchange_interface.execute_trade.assert_awaited_once()
    order_manager.db_queries.store_order.assert_not_awaited()
    order_manager.logger.error.assert_called_with(
        "Failed to place order for BTC/USDT: Exchange Error"
    )


@pytest.mark.asyncio
async def test_cancel_order_success(order_manager):
    """Test successful order cancellation."""
    order_manager.exchange_interface.cancel_trade.return_value = True
    order_manager.db_queries.update_order_status.return_value = True

    result = await order_manager.cancel_order('order002')
    assert result == True
    order_manager.exchange_interface.cancel_trade.assert_awaited_once_with('order002')
    order_manager.db_queries.update_order_status.assert_awaited_once_with('order002', 'cancelled')


@pytest.mark.asyncio
async def test_cancel_order_failure(order_manager):
    """Test order cancellation when exchange fails."""
    order_manager.exchange_interface.cancel_trade.return_value = False

    result = await order_manager.cancel_order('order003')
    assert result == False
    order_manager.exchange_interface.cancel_trade.assert_awaited_once_with('order003')
    order_manager.db_queries.update_order_status.assert_not_awaited()
    order_manager.logger.error.assert_called_with(
        "Failed to cancel order order003: Exchange failed to cancel."
    )


@pytest.mark.asyncio
async def test_get_order_status_success(order_manager):
    """Test successful retrieval of order status."""
    mock_status = {'id': 'order004', 'status': 'filled'}
    order_manager.exchange_interface.get_order_status.return_value = mock_status

    status = await order_manager.get_order_status('order004')
    assert status == mock_status
    order_manager.exchange_interface.get_order_status.assert_awaited_once_with('order004')


@pytest.mark.asyncio
async def test_get_order_status_exchange_error(order_manager):
    """Test retrieval of order status when exchange raises an error."""
    order_manager.exchange_interface.get_order_status.side_effect = ExchangeError("Exchange Error")

    status = await order_manager.get_order_status('order005')
    assert status is None
    order_manager.exchange_interface.get_order_status.assert_awaited_once_with('order005')
    order_manager.logger.error.assert_called_with(
        "Failed to retrieve status for order005: Exchange Error"
    ) 