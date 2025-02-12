import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from execution.main_loop import MainLoop
from execution.exchange_interface import ExchangeInterface
from risk.manager import RiskManager
from risk.limits import RiskLimits
from database.queries import DatabaseQueries
from utils.logger import setup_logging
from utils.health_monitor import HealthMonitor


@pytest.fixture
def mock_exchange_interface():
    """Provide a mocked ExchangeInterface."""
    interface = MagicMock(spec=ExchangeInterface)
    interface.fetch_market_data = AsyncMock(return_value={'BTC/USDT': {'price': Decimal('50000')}})
    interface.place_trade = AsyncMock(return_value={'success': True, 'order_id': 'order130'})
    interface.execute_trade = AsyncMock()
    return interface


@pytest.fixture
def mock_risk_manager():
    """Provide a mocked RiskManager."""
    manager = MagicMock(spec=RiskManager)
    manager.validate_market_conditions.return_value = True
    manager.calculate_risk_metrics.return_value = True
    return manager


@pytest.fixture
def db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def main_loop(mock_exchange_interface, mock_risk_manager, db_queries, logger):
    """Provide a MainLoop instance."""
    return MainLoop(
        exchange_interface=mock_exchange_interface,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )


@pytest.mark.asyncio
async def test_full_trading_cycle_success(main_loop):
    """Test a full trading cycle with successful operations."""
    # Mock trade placement
    main_loop.exchange_interface.place_trade.return_value = {'success': True, 'order_id': 'order131'}
    main_loop.db_queries.log_trade.return_value = True

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        # Run a single cycle
        await main_loop.run_cycle()
        main_loop.exchange_interface.fetch_market_data.assert_awaited_once()
        main_loop.risk_manager.validate_market_conditions.assert_awaited_once()
        main_loop.exchange_interface.place_trade.assert_awaited_once()
        main_loop.db_queries.log_trade.assert_awaited_once_with('order131', {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': Decimal('1'), 'price': Decimal('50000')})
        mock_sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_full_trading_cycle_risk_failure(main_loop):
    """Test a full trading cycle where risk validation fails."""
    main_loop.risk_manager.validate_market_conditions.return_value = False

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        await main_loop.run_cycle()
        main_loop.exchange_interface.fetch_market_data.assert_awaited_once()
        main_loop.risk_manager.validate_market_conditions.assert_awaited_once()
        main_loop.exchange_interface.place_trade.assert_not_awaited()
        main_loop.db_queries.log_trade.assert_not_awaited()
        mock_sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_full_trading_cycle_trade_failure(main_loop):
    """Test a full trading cycle where trade placement fails."""
    main_loop.exchange_interface.place_trade.return_value = {'success': False, 'error': 'Trade Failed'}

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        await main_loop.run_cycle()
        main_loop.exchange_interface.fetch_market_data.assert_awaited_once()
        main_loop.risk_manager.validate_market_conditions.assert_awaited_once()
        main_loop.exchange_interface.place_trade.assert_awaited_once()
        main_loop.db_queries.log_trade.assert_not_awaited()
        main_loop.logger.error.assert_called_with("Trade placement failed: Trade Failed")
        mock_sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_full_trading_cycle_fetch_market_data_failure(main_loop):
    """Test a full trading cycle where fetching market data fails."""
    main_loop.exchange_interface.fetch_market_data.side_effect = Exception("Market Data Fetch Error")

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        await main_loop.run_cycle()
        main_loop.exchange_interface.fetch_market_data.assert_awaited_once()
        main_loop.risk_manager.validate_market_conditions.assert_not_awaited()
        main_loop.exchange_interface.place_trade.assert_not_awaited()
        main_loop.db_queries.log_trade.assert_not_awaited()
        main_loop.logger.error.assert_called_with("Error in MainLoop.execute_cycle: Market Data Fetch Error")
        mock_sleep.assert_awaited_once_with(1) 