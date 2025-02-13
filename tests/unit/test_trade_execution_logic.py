import logging
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.execution.exchange_interface import ExchangeInterface
from src.exchanges.exchange_manager import ExchangeManager
from src.risk.manager import RiskManager
from src.risk.limits import RiskLimits
from src.database.queries import DatabaseQueries
from src.utils.error_handler import ExchangeError, RiskError


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
async def test_execute_trade_within_risk_limits(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test executing a trade that is within risk limits."""
    # Mock risk validation to pass
    mock_risk_manager.validate_trade = AsyncMock(return_value=True)
    
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Mock successful order creation
    mock_order = {
        'id': 'order128',
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
    assert result['order_id'] == 'order128'
    mock_exchange_manager.exchange.create_order.assert_awaited_once()
    mock_risk_manager.validate_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_trade_exceeds_risk_limits(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test executing a trade that exceeds risk limits."""
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
        symbol='ETH/USDT',
        side='sell',
        amount=Decimal('10'),
        order_type='market'
    )
    
    assert result['success'] is False
    assert result['error'] == "Trade validation failed."
    mock_exchange_manager.exchange.create_order.assert_not_awaited()
    mock_risk_manager.validate_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_trade_exchange_failure(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test executing a trade when exchange raises an error."""
    # Mock risk validation to pass
    mock_risk_manager.validate_trade = AsyncMock(return_value=True)
    
    # Mock exchange error
    mock_exchange_manager.exchange.create_order.side_effect = ExchangeError("Order Failed")
    
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    result = await exchange_interface.execute_trade(
        symbol='ETH/USDT',
        side='buy',
        amount=Decimal('5'),
        order_type='limit',
        price=Decimal('3000')
    )
    
    assert result['success'] is False
    assert result['error'] == "Order Failed"
    mock_exchange_manager.exchange.create_order.assert_awaited_once()
    mock_risk_manager.validate_trade.assert_awaited_once() 