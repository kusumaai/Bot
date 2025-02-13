import logging
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
import asyncio

from trading.ratchet import RatchetManager
from execution.exchange_interface import ExchangeInterface
from risk.limits import RiskLimits
from database.queries import DatabaseQueries
from utils.exceptions import RatchetError


@pytest.fixture
def risk_limits():
    """Provide test risk limits."""
    return RiskLimits.from_config({
        'max_position_size': '0.1',
        'min_position_size': '0.01',
        'max_positions': 3,
        'max_leverage': '2.0',
        'max_drawdown': '0.1',
        'max_daily_loss': '0.03',
        'emergency_stop_pct': '3.0',
        'risk_factor': '0.01',
        'kelly_scaling': '0.5',
        'max_correlation': '0.7',
        'max_sector_exposure': '0.3',
        'max_volatility': '0.05',
        'min_liquidity': '100000'
    })


@pytest.fixture
def mock_exchange_interface():
    """Provide a mocked ExchangeInterface."""
    interface = MagicMock(spec=ExchangeInterface)
    interface.execute_trade = AsyncMock()
    return interface


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
def ratchet_manager(mock_exchange_interface, db_queries, risk_limits, logger):
    """Provide a RatchetManager instance."""
    ctx = MagicMock()
    ctx.exchange_interface = mock_exchange_interface
    ctx.db_queries = db_queries
    ctx.risk_manager.risk_limits = risk_limits
    ctx.logger = logger
    ratchet_mgr = RatchetManager(ctx)
    asyncio.run(ratchet_mgr.initialize())
    return ratchet_mgr


@pytest.mark.asyncio
async def test_ratchet_initialize_trade(ratchet_manager):
    """Test initializing a trade in RatchetManager."""
    trade_id = "trade123"
    entry_price = 50000.0
    take_profit = 51000.0
    stop_loss = 49000.0
    
    await ratchet_manager.initialize_trade(trade_id, entry_price, take_profit, stop_loss)
    
    assert trade_id in ratchet_manager.trades
    trade = ratchet_manager.trades[trade_id]
    assert trade['entry_price'] == entry_price
    assert trade['take_profit'] == take_profit
    assert trade['stop_loss'] == stop_loss
    assert trade['current_level'] == 0


@pytest.mark.asyncio
async def test_ratchet_increase_position_success(ratchet_manager):
    """Test successful ratchet increase of a position."""
    position = {
        'id': 'pos001',
        'symbol': 'BTC/USDT',
        'direction': 'long',
        'size': Decimal('0.1'),
        'entry_price': Decimal('50000'),
        'status': 'active'
    }
    ratchet_manager.exchange_interface.execute_trade.return_value = {'success': True, 'order_id': 'ratchet_order001'}

    result = await ratchet_manager.increase_position(position, Decimal('0.02'))
    assert result == {'success': True, 'order_id': 'ratchet_order001'}
    ratchet_manager.exchange_interface.execute_trade.assert_awaited_once_with(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.02'),
        order_type='limit',
        price=Decimal('50000') * Decimal('1.05')  # 5% higher for ratchet
    )
    ratchet_manager.db_queries.update_position_size.assert_awaited_once_with('pos001', Decimal('0.12'))


@pytest.mark.asyncio
async def test_ratchet_decrease_position_success(ratchet_manager):
    """Test successful ratchet decrease of a position."""
    position = {
        'id': 'pos002',
        'symbol': 'ETH/USDT',
        'direction': 'short',
        'size': Decimal('2'),
        'entry_price': Decimal('3000'),
        'status': 'active'
    }
    ratchet_manager.exchange_interface.execute_trade.return_value = {'success': True, 'order_id': 'ratchet_order002'}

    result = await ratchet_manager.decrease_position(position, Decimal('0.05'))
    assert result == {'success': True, 'order_id': 'ratchet_order002'}
    ratchet_manager.exchange_interface.execute_trade.assert_awaited_once_with(
        symbol='ETH/USDT',
        side='buy',
        amount=Decimal('0.05'),
        order_type='limit',
        price=Decimal('3000') * Decimal('0.95')  # 5% lower for ratchet
    )
    ratchet_manager.db_queries.update_position_size.assert_awaited_once_with('pos002', Decimal('1.95'))


@pytest.mark.asyncio
async def test_ratchet_increase_position_exchange_error(ratchet_manager):
    """Test ratchet increase when exchange raises an error."""
    position = {
        'id': 'pos003',
        'symbol': 'SOL/USDT',
        'direction': 'long',
        'size': Decimal('1'),
        'entry_price': Decimal('100'),
        'status': 'active'
    }
    ratchet_manager.exchange_interface.execute_trade.side_effect = RatchetError("Exchange Failed")

    with pytest.raises(RatchetError, match="Failed to increase position pos003: Exchange Failed"):
        await ratchet_manager.increase_position(position, Decimal('0.02'))
    ratchet_manager.exchange_interface.execute_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_ratchet_decrease_position_exchange_error(ratchet_manager):
    """Test ratchet decrease when exchange raises an error."""
    position = {
        'id': 'pos004',
        'symbol': 'ADA/USDT',
        'direction': 'short',
        'size': Decimal('1.5'),
        'entry_price': Decimal('2.0'),
        'status': 'active'
    }
    ratchet_manager.exchange_interface.execute_trade.side_effect = RatchetError("Exchange Failed")

    with pytest.raises(RatchetError, match="Failed to decrease position pos004: Exchange Failed"):
        await ratchet_manager.decrease_position(position, Decimal('0.05'))
    ratchet_manager.exchange_interface.execute_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_ratchet_max_steps_reached(ratchet_manager):
    """Test that ratchet does not exceed maximum steps."""
    ratchet_manager.ratchet_steps = ratchet_manager.risk_limits.max_ratchet_steps
    position = {
        'id': 'pos005',
        'symbol': 'BTC/USDT',
        'direction': 'long',
        'size': Decimal('0.2'),
        'entry_price': Decimal('55000'),
        'status': 'active'
    }

    with pytest.raises(RatchetError, match="Maximum ratchet steps reached for position pos005"):
        await ratchet_manager.increase_position(position, Decimal('0.02'))
    ratchet_manager.exchange_interface.execute_trade.assert_not_awaited()
    ratchet_manager.db_queries.update_position_size.assert_not_awaited() 