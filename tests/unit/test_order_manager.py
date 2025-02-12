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
async def test_store_order_success(order_manager):
    """Test successful storage of an order."""
    order = {
        'id': 'order127',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': Decimal('0.5'),
        'price': Decimal('50000'),
        'status': 'open'
    }
    order_manager.db_queries.store_order.return_value = True

    result = await order_manager.store_order(order)
    assert result is True
    order_manager.db_queries.store_order.assert_awaited_once_with(order)


@pytest.mark.asyncio
async def test_store_order_failure(order_manager):
    """Test storage of an order failure due to database error."""
    order = {
        'id': 'order128',
        'symbol': 'ETH/USDT',
        'side': 'sell',
        'amount': Decimal('1'),
        'price': Decimal('3000'),
        'status': 'open'
    }
    order_manager.db_queries.store_order.side_effect = Exception("DB Error")

    with pytest.raises(OrderError, match="Failed to store order order128: DB Error"):
        await order_manager.store_order(order)
    order_manager.db_queries.store_order.assert_awaited_once_with(order)


@pytest.mark.asyncio
async def test_place_order_success(order_manager):
    """Test successful place_order method."""
    mock_order_response = {'success': True, 'order_id': 'order129'}
    mock_trade = {'id': 'order129', 'status': 'open'}
    order_manager.exchange_interface.execute_trade.return_value = mock_order_response
    order_manager.db_queries.store_order.return_value = True

    result = await order_manager.place_order(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.1'),
        order_type='limit',
        price=Decimal('50000')
    )
    assert result is True
    order_manager.exchange_interface.execute_trade.assert_awaited_once()
    order_manager.db_queries.store_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_place_order_exchange_failure(order_manager):
    """Test place_order when exchange raises an error."""
    order_manager.exchange_interface.execute_trade.side_effect = ExchangeError("Exchange Failed")

    result = await order_manager.place_order(
        symbol='ETH/USDT',
        side='sell',
        amount=Decimal('1'),
        order_type='market',
        price=Decimal('3000')
    )
    assert result is False
    order_manager.exchange_interface.execute_trade.assert_awaited_once()
    order_manager.db_queries.store_order.assert_not_awaited()
    order_manager.logger.error.assert_called_with(
        "Failed to place order for ETH/USDT: Exchange Failed"
    )


@pytest.mark.asyncio
async def test_place_order_risk_failure(order_manager):
    """Test place_order when risk validation fails."""
    order_manager.exchange_interface.execute_trade.return_value = {'success': False, 'error': 'Risk Validation Failed'}

    result = await order_manager.place_order(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.2'),
        order_type='limit',
        price=Decimal('50000')
    )
    assert result is False
    order_manager.exchange_interface.execute_trade.assert_awaited_once()
    order_manager.db_queries.store_order.assert_not_awaited()
    order_manager.logger.error.assert_called_with(
        "Failed to place order for BTC/USDT: Risk Validation Failed"
    ) 