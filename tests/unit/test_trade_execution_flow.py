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
async def test_full_trade_execution_flow(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test the complete trade execution flow from validation to order creation."""
    # Mock risk validation
    mock_risk_manager.validate_trade = AsyncMock(return_value=True)
    
    # Mock successful order creation
    mock_order = {
        'id': 'order129',
        'symbol': 'BTC/USDT',
        'status': 'open',
        'price': '50000',
        'amount': '0.1'
    }
    mock_exchange_manager.exchange.create_order.return_value = mock_order
    
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )
    
    # Execute trade
    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.1'),
        order_type='limit',
        price=Decimal('50000')
    )
    
    # Assertions
    assert result['success'] is True
    assert result['order_id'] == 'order129'
    mock_risk_manager.validate_trade.assert_awaited_once()
    mock_exchange_manager.exchange.create_order.assert_awaited_once()
    db_queries.store_trade.assert_awaited_once() 