import pytest
from decimal import Decimal
import logging
from unittest.mock import AsyncMock, MagicMock

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
        sandbox=True
    )
    manager.exchange = AsyncMock()
    return manager


@pytest.fixture
def mock_risk_manager():
    """Provide a mocked RiskManager."""
    mock_risk = MagicMock(spec=RiskManager)
    mock_risk.validate_trade = AsyncMock(return_value=(True, None))
    return mock_risk


@pytest.fixture
def db_queries():
    """Provide a mocked DatabaseQueries."""
    mock_db = AsyncMock(spec=DatabaseQueries)
    mock_db.log_trade = AsyncMock()
    return mock_db


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def exchange_interface(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Provide an ExchangeInterface instance with mocked dependencies."""
    ctx = MagicMock()
    ctx.logger = logger
    ctx.risk_manager = mock_risk_manager
    ctx.db_queries = db_queries
    ctx.config = {
        "exchange_id": "binance",
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "paper_mode": True,
        "database": {"path": "data/test_trading.db"},
        "initial_balance": "10000",
        "rate_limit_per_second": 5
    }
    
    exchange_interface = ExchangeInterface(ctx)
    exchange_interface.exchange_manager.exchange.create_order = AsyncMock(return_value={
        'id': 'order_test',
        'symbol': 'BTC/USDT',
        'status': 'open',
        'price': '50000',
        'amount': '0.1'
    })
    return exchange_interface


@pytest.mark.asyncio
async def test_execute_trade_within_risk_limits(exchange_interface, mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test executing a trade that is within risk limits."""
    mock_risk_manager.validate_trade = AsyncMock(return_value=(True, None))
    
    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.1'),
        order_type='limit',
        price=Decimal('50000')
    )
    
    assert result['success'] is True
    assert result['order_id'] == 'order_test'
    mock_exchange_manager.exchange.create_order.assert_awaited_once()
    mock_risk_manager.validate_trade.assert_awaited_once()
    db_queries.log_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_trade_exceeds_risk_limits(exchange_interface, mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test executing a trade that exceeds risk limits."""
    mock_risk_manager.validate_trade = AsyncMock(return_value=(False, "Trade validation failed."))
    
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
    db_queries.log_trade.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_trade_exchange_failure(exchange_interface, mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Test executing a trade when exchange raises an error."""
    mock_risk_manager.validate_trade = AsyncMock(return_value=(True, None))
    mock_exchange_manager.exchange.create_order.side_effect = ExchangeError("Order Failed")
    
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
    db_queries.log_trade.assert_not_awaited() 