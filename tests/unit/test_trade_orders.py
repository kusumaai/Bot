import pytest
from decimal import Decimal
import logging
from unittest.mock import AsyncMock, MagicMock

from src.execution.exchange_interface import ExchangeInterface
from src.exchanges.exchange_manager import ExchangeManager
from src.risk.manager import RiskManager
from src.database.queries import DatabaseQueries
from src.utils.error_handler import ExchangeError

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


@pytest.fixture
def mock_risk_manager():
    """Provide a mocked RiskManager."""
    return AsyncMock(spec=RiskManager)


@pytest.mark.asyncio
async def test_create_order_success(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test successful creation of an order."""
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Mock create_order
    order = {
        'id': 'order126',
        'symbol': 'BTC/USDT',
        'status': 'open',
        'price': '50000',
        'amount': '0.1'
    }
    mock_exchange_manager.exchange.create_order.return_value = order
    
    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.1'),
        order_type='limit',
        price=Decimal('50000')
    )
    
    assert result['success'] is True
    assert result['order_id'] == 'order126'
    mock_exchange_manager.exchange.create_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_order_api_failure(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test creation of an order when the exchange API fails."""
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Mock create_order to raise an ExchangeError
    mock_exchange_manager.exchange.create_order.side_effect = ExchangeError("API Failure")
    
    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='sell',
        amount=Decimal('0.1'),
        order_type='market'
    )
    
    assert result['success'] is False
    assert result['error'] == "API Failure"
    mock_exchange_manager.exchange.create_order.assert_awaited_once()


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