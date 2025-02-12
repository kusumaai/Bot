import pytest
from decimal import Decimal
import logging
from unittest.mock import AsyncMock, MagicMock

from trading.ratchet import RatchetManager
from risk.limits import RiskLimits
from execution.exchange_interface import ExchangeInterface
from utils.exceptions import RatchetError
from typing import Dict, Any


@pytest.fixture
def risk_limits():
    """Provide test risk limits."""
    return RiskLimits.from_config({
        'max_leverage': '2.0',
        'max_drawdown': '0.1',
        'max_daily_loss': '0.03'
    })


@pytest.fixture
def mock_exchange_interface():
    mock = AsyncMock(spec=ExchangeInterface)
    mock.fetch_ticker = AsyncMock(return_value='50000')
    mock.close_position = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def ratchet_manager():
    mock_ctx = MagicMock()
    mock_ctx.logger = logging.getLogger("test")
    return RatchetManager(mock_ctx)


@pytest.mark.asyncio
async def test_ratchet_increase_position_success(ratchet_manager_fixture):
    """Test successful ratchet increase of a position."""
    position = {
        'id': 'pos001',
        'symbol': 'BTC/USDT',
        'direction': 'long',
        'size': Decimal('0.1'),
        'entry_price': Decimal('50000'),
        'status': 'active'
    }
    
    result = await ratchet_manager_fixture.increase_position(position, Decimal('5000'))
    assert result == {'success': True, 'order_id': 'ratchet_order001'}
    ratchet_manager_fixture.exchange_interface.execute_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_ratchet_increase_position_exceeds_limits(ratchet_manager_fixture):
    """Test ratchet increase that exceeds risk limits."""
    position = {
        'id': 'pos002',
        'symbol': 'ETH/USDT',
        'direction': 'short',
        'size': Decimal('2.0'),
        'entry_price': Decimal('3000'),
        'status': 'active'
    }
    
    with pytest.raises(RatchetError, match="Increasing position exceeds maximum leverage"):
        await ratchet_manager_fixture.increase_position(position, Decimal('10000'))


@pytest.mark.asyncio
async def test_ratchet_decrease_position_success(ratchet_manager_fixture):
    """Test successful ratchet decrease of a position."""
    position = {
        'id': 'pos003',
        'symbol': 'SOL/USDT',
        'direction': 'long',
        'size': Decimal('1.0'),
        'entry_price': Decimal('100'),
        'status': 'active'
    }
    
    # Mock execute_trade for decrease
    ratchet_manager_fixture.exchange_interface.execute_trade.return_value = {'success': True, 'order_id': 'ratchet_order002'}
    
    result = await ratchet_manager_fixture.decrease_position(position, Decimal('500'))
    assert result == {'success': True, 'order_id': 'ratchet_order002'}
    ratchet_manager_fixture.exchange_interface.execute_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_ratchet_decrease_position_failure(ratchet_manager_fixture):
    """Test ratchet decrease failure due to exchange error."""
    position = {
        'id': 'pos004',
        'symbol': 'ADA/USDT',
        'direction': 'short',
        'size': Decimal('1.5'),
        'entry_price': Decimal('2.0'),
        'status': 'active'
    }
    
    # Mock execute_trade to fail
    ratchet_manager_fixture.exchange_interface.execute_trade.return_value = {'success': False, 'error': 'Insufficient Balance'}
    
    with pytest.raises(RatchetError, match="Failed to execute ratchet decrease"):
        await ratchet_manager_fixture.decrease_position(position, Decimal('500'))
    ratchet_manager_fixture.exchange_interface.execute_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_trade(ratchet_manager):
    await ratchet_manager.initialize_trade("trade123", Decimal("50000"), "BTCUSD")
    normalized_id = ratchet_manager._normalize_trade_id("trade123", "BTCUSD")
    assert normalized_id in ratchet_manager.active_trades


@pytest.mark.asyncio
async def test_update_position_ratchet(ratchet_manager, mock_exchange_interface):
    await ratchet_manager.initialize_trade("trade123", Decimal("50000"), "BTCUSD")
    updated_stop = await ratchet_manager.update_position_ratchet("BTCUSD", Decimal("50500"), {})
    assert updated_stop == Decimal("50400")


@pytest.mark.asyncio
async def test_monitor_trades(ratchet_manager, mock_exchange_interface):
    await ratchet_manager.initialize_trade("trade123", Decimal("50000"), "BTCUSD")
    ratchet_manager.update_position_ratchet = AsyncMock(return_value=Decimal("50400"))
    await ratchet_manager.monitor_trades(mock_exchange_interface)
    mock_exchange_interface.close_position.assert_called_with("BTCUSD", 0) 