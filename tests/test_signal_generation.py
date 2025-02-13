from decimal import Decimal
import pytest
from signals.ga_synergy import generate_ga_signals, GASignal
from utils.exceptions import InvalidOrderError

def test_generate_ga_signals_valid():
    data = {
        "symbol": "BTCUSD",
        "action": "buy",
        "price": "50000",
        "quantity": "1"
    }
    signal = generate_ga_signals(data)
    assert isinstance(signal, GASignal)
    assert signal.symbol == "BTCUSD"
    assert signal.action == "buy"
    assert signal.price == Decimal("50000")
    assert signal.quantity == Decimal("1")

def test_generate_ga_signals_invalid_action():
    data = {
        "symbol": "BTCUSD",
        "action": "hold",
        "price": "50000",
        "quantity": "1"
    }
    with pytest.raises(InvalidOrderError):
        generate_ga_signals(data) 