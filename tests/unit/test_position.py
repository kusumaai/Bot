import pytest
from decimal import Decimal
from datetime import datetime

from trading.position import Position
from utils.error_handler import PositionError


@pytest.mark.asyncio
async def test_position_creation_valid():
    """Test creation of a Position with valid parameters."""
    position = Position(
        symbol='BTC/USDT',
        side='long',
        size=Decimal('0.1'),
        entry_price=Decimal('50000'),
        timestamp=int(datetime.now().timestamp())
    )
    assert position.symbol == 'BTC/USDT'
    assert position.side == 'long'
    assert position.size == Decimal('0.1')
    assert position.entry_price == Decimal('50000')
    assert position.status == 'active'


@pytest.mark.asyncio
async def test_position_creation_invalid_side():
    """Test creation of a Position with invalid side."""
    with pytest.raises(ValueError, match="Invalid side: sideways"):
        Position(
            symbol='BTC/USDT',
            side='sideways',
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            timestamp=int(datetime.now().timestamp())
        )


@pytest.mark.asyncio
async def test_position_update_price_valid():
    """Test updating the price of an active Position."""
    position = Position(
        symbol='ETH/USDT',
        side='short',
        size=Decimal('0.5'),
        entry_price=Decimal('3000'),
        timestamp=int(datetime.now().timestamp())
    )
    
    await position.update_price(Decimal('2950'))
    assert position.current_price == Decimal('2950')
    assert position.unrealized_pnl == Decimal('250')  # (entry_price - current_price) * size


@pytest.mark.asyncio
async def test_position_update_price_invalid():
    """Test updating the price of a Position with invalid value."""
    position = Position(
        symbol='ETH/USDT',
        side='short',
        size=Decimal('0.5'),
        entry_price=Decimal('3000'),
        timestamp=int(datetime.now().timestamp())
    )
    
    with pytest.raises(PositionError, match="Invalid price update"):
        await position.update_price(Decimal('-2950'))


@pytest.mark.asyncio
async def test_position_close_success():
    """Test successfully closing a Position."""
    position = Position(
        symbol='SOL/USDT',
        side='long',
        size=Decimal('1.0'),
        entry_price=Decimal('100'),
        timestamp=int(datetime.now().timestamp())
    )
    
    await position.close(Decimal('110'))
    assert position.status == 'closed'
    assert position.exit_price == Decimal('110')
    assert position.realized_pnl == Decimal('100')


@pytest.mark.asyncio
async def test_position_close_already_closed():
    """Test closing an already closed Position."""
    position = Position(
        symbol='SOL/USDT',
        side='long',
        size=Decimal('1.0'),
        entry_price=Decimal('100'),
        timestamp=int(datetime.now().timestamp())
    )
    
    await position.close(Decimal('110'))
    with pytest.raises(PositionError, match="Position already closed"):
        await position.close(Decimal('115')) 