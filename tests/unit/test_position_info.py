import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.trading.position_info import PositionInfo


def test_position_info_initialization():
    """Test initialization of PositionInfo."""
    pos_info = PositionInfo(
        id='pos001',
        symbol='BTC/USDT',
        direction='long',
        size=Decimal('0.1'),
        entry_price=Decimal('50000'),
        timestamp=int(datetime.now(tz=timezone.utc).timestamp())
    )
    
    assert pos_info.id == 'pos001'
    assert pos_info.symbol == 'BTC/USDT'
    assert pos_info.direction == 'long'
    assert pos_info.size == Decimal('0.1')
    assert pos_info.entry_price == Decimal('50000')
    assert pos_info.status == 'active'


def test_position_info_update_pnl():
    """Test updating PnL in PositionInfo."""
    pos_info = PositionInfo(
        id='pos002',
        symbol='ETH/USDT',
        direction='short',
        size=Decimal('10'),
        entry_price=Decimal('3000'),
        timestamp=int(datetime.now(tz=timezone.utc).timestamp())
    )
    
    pos_info.update_pnl(Decimal('500'))
    assert pos_info.unrealized_pnl == Decimal('500')


def test_position_info_close():
    """Test closing a PositionInfo."""
    pos_info = PositionInfo(
        id='pos003',
        symbol='SOL/USDT',
        direction='long',
        size=Decimal('1'),
        entry_price=Decimal('100'),
        timestamp=int(datetime.now(tz=timezone.utc).timestamp())
    )
    
    pos_info.close(Decimal('110'), Decimal('1000'), timestamp=int(datetime.now(tz=timezone.utc).timestamp()))
    assert pos_info.status == 'closed'
    assert pos_info.exit_price == Decimal('110')
    assert pos_info.realized_pnl == Decimal('1000')
    assert pos_info.exit_timestamp is not None 


def test_position_info_creation():
    info = PositionInfo(
        symbol="BTCUSD",
        current_price=Decimal("51000"),
        unrealized_pnl=Decimal("1000"),
        realized_pnl=Decimal("500")
    )
    assert info.symbol == "BTCUSD"
    assert info.current_price == Decimal("51000")
    assert info.unrealized_pnl == Decimal("1000")
    assert info.realized_pnl == Decimal("500") 