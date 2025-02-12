import pytest
from decimal import Decimal
import logging
from unittest.mock import AsyncMock

from execution.exchange_interface import ExchangeInterface
from exchanges.exchange_manager import ExchangeManager
from risk.manager import RiskManager
from database.queries import DatabaseQueries
from utils.error_handler import ExchangeError


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
async def test_execute_trade_with_risk_validation(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test trade execution with successful risk validation."""
    # Mock risk validation to pass
    mock_risk_manager.validate_trade = AsyncMock(return_value=True)
    
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Mock successful order execution
    mock_order = {
        'id': 'order125',
        'symbol': 'BTC/USDT',
        'status': 'open',
        'price': '50000',
        'amount': '0.1'
    }
    mock_exchange_manager.exchange.create_order.return_value = mock_order
    
    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.1'),
        order_type='limit',
        price=Decimal('50000')
    )
    
    assert result['success'] is True
    assert result['order_id'] == 'order125'
    mock_risk_manager.validate_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_trade_with_risk_validation_failure(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test trade execution when risk validation fails."""
    # Mock risk validation to fail
    mock_risk_manager.validate_trade = AsyncMock(return_value=False)
    
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Attempt to execute trade
    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.1'),
        order_type='limit',
        price=Decimal('50000')
    )
    
    assert result['success'] is False
    assert result['error'] == "Trade validation failed."
    mock_exchange_manager.exchange.create_order.assert_not_awaited()
    mock_risk_manager.validate_trade.assert_awaited_once() 