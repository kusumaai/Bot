from decimal import Decimal
import time
import pytest
from trading.position import Position
from utils.exceptions import PositionError

def test_position_creation():
    position = Position(
        symbol="BTCUSD",
        side="buy",
        entry_price=Decimal("50000"),
        size=Decimal("1"),
        timestamp=int(time.time() * 1000)
    )
    assert position.symbol == "BTCUSD"
    assert position.side == "buy"
    assert position.entry_price == Decimal("50000")
    assert position.size == Decimal("1")

def test_position_invalid_creation():
    with pytest.raises(ValueError):
        Position(
            symbol="",
            side="buy",
            entry_price=Decimal("-50000"),
            size=Decimal("1"),
            timestamp=int(time.time() * 1000)
        ) 