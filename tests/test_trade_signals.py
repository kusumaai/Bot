import pytest
from signals.ga_synergy import generate_ga_signals, GASignal
from utils.exceptions import InvalidOrderError
from decimal import Decimal
from typing import Dict, Any

def test_trade_signals_valid():
    data = {
        "symbol": "ETHUSD",
        "action": "sell",
        "price": "4000",
        "quantity": "2"
    }
    signal = generate_ga_signals(data)
    assert signal.symbol == "ETHUSD"
    assert signal.action == "sell"
    assert signal.price == Decimal("4000")
    assert signal.quantity == Decimal("2")

def test_trade_signals_invalid_action():
    data = {
        "symbol": "ETHUSD",
        "action": "hold",
        "price": "4000",
        "quantity": "2"
    }
    with pytest.raises(InvalidOrderError):
        generate_ga_signals(data) 