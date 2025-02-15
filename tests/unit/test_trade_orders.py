import pytest
from decimal import Decimal
import logging
from unittest.mock import AsyncMock, MagicMock

from src.execution.exchange_interface import ExchangeInterface
from src.exchanges.exchange_manager import ExchangeManager
from src.risk.manager import RiskManager
from src.database.queries import DatabaseQueries
from src.utils.error_handler import ExchangeError
from src.execution.order_manager import OrderManager
from src.utils.exceptions import OrderStoreError, OrderCancelError

#mock exchange manager for the trade orders tests
@pytest.fixture
def mock_exchange_manager():
    """Provide a mocked ExchangeManager."""
    manager = ExchangeManager(
        exchange_id='binance',
        api_key='test_key',
        api_secret='test_secret',
        logger=logging.getLogger("TestExchangeManager"),
        db_queries=AsyncMock(spec=DatabaseQueries)
    )
    manager.exchange = AsyncMock()
    return manager

#mock risk manager for the trade orders tests
@pytest.fixture
def mock_risk_manager():
    """Provide a mocked RiskManager."""
    return AsyncMock(spec=RiskManager)

#mock database queries for the trade orders tests
@pytest.fixture
def db_queries():
    return MagicMock(spec=DatabaseQueries)

@pytest.fixture
def logger():
    return MagicMock()

@pytest.fixture
def order_manager(db_queries, logger):
    return OrderManager(db_queries=db_queries, logger=logger)

@pytest.mark.asyncio
async def test_create_order_success(order_manager, db_queries):
    """Test successful order creation."""
    order_details = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': Decimal('0.1'),
        'price': Decimal('50000')
    }
    db_queries.insert_order = MagicMock(return_value=True)
    result = await order_manager.store_order(order_details)
    assert result is True
    db_queries.insert_order.assert_called_once_with(order_details)

#test create order api failure for the trade orders tests
@pytest.mark.asyncio
async def test_create_order_api_failure(order_manager, db_queries, logger):
    """Test order creation failure due to API error."""
    order_details = {
        'symbol': 'BTC/USDT',
        'side': 'sell',
        'amount': Decimal('0.2'),
        'price': Decimal('48000')
    }
    db_queries.insert_order = MagicMock(side_effect=Exception("API Failure"))
    with pytest.raises(OrderStoreError, match="Error storing order: API Failure"):
        await order_manager.store_order(order_details)
    db_queries.insert_order.assert_called_once_with(order_details)

#test cancel order success for the trade orders tests
@pytest.mark.asyncio
async def test_cancel_order_success(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test successful cancellation of an existing order."""
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Mock close_order
    closed_order = {
        'id': 'order127',
        'symbol': 'BTC/USDT',
        'status': 'closed',
        'price': '50000',
        'amount': '0.1'
    }
    mock_exchange_manager.exchange.close_order.return_value = closed_order
    
    success = await exchange_interface.cancel_trade('order127')
    assert success is True
    mock_exchange_manager.exchange.close_order.assert_awaited_once_with('order127')

#test cancel order not found for the trade orders tests
@pytest.mark.asyncio
async def test_cancel_order_not_found(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test cancellation of a non-existent order."""
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Mock close_order returning None
    mock_exchange_manager.exchange.close_order.return_value = None
    
    success = await exchange_interface.cancel_trade('invalid_order')
    assert success is False
    mock_exchange_manager.exchange.close_order.assert_awaited_once_with('invalid_order') 

#test cancel order api failure for the trade orders tests
@pytest.mark.asyncio
async def test_cancel_order_api_failure(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test cancellation failure due to API error."""
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Mock close_order raising an exception
    mock_exchange_manager.exchange.close_order.side_effect = Exception("API Failure")
    
    with pytest.raises(OrderCancelError, match="Error cancelling order: API Failure"):
        await exchange_interface.cancel_trade('order127')
    mock_exchange_manager.exchange.close_order.assert_awaited_once_with('order127')

#test cancel order not found for the trade orders tests
@pytest.mark.asyncio
async def test_cancel_order_not_found(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test cancellation of a non-existent order."""
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )   
    # Mock close_order returning None
    mock_exchange_manager.exchange.close_order.return_value = None
    
    success = await exchange_interface.cancel_trade('invalid_order')
    assert success is False
    mock_exchange_manager.exchange.close_order.assert_awaited_once_with('invalid_order')        


