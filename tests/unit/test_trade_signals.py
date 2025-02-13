import pytest
from decimal import Decimal

from src.signals.ga_synergy import generate_ga_signals
from src.signals.ml_signals import generate_ml_signals
from src.utils.error_handler import ValidationError

def test_generate_ml_signals():
    """Test generation of ML-based trade signals."""
    input_data = {
        'symbol': 'BTC/USDT',
        'trend': 'bullish',
        'strength': Decimal('0.8')
    }
    
    signals = generate_ml_signals(input_data)
    assert isinstance(signals, dict)
    assert 'signal' in signals
    assert signals['signal'] == 'buy'  # Assuming bullish trend leads to buy signal


def test_generate_ga_signals():
    """Test generation of GA-based trade signals."""
    input_data = {
        'symbol': 'ETH/USDT',
        'strategy': 'crossover',
        'indicator_values': {'macd': Decimal('0.5'), 'rsi': Decimal('30')}
    }
    
    signals = generate_ga_signals(input_data)
    assert isinstance(signals, dict)
    assert 'signal' in signals
    assert signals['signal'] == 'buy'  # Assuming crossover strategy leads to buy signal


def test_invalid_signal_data():
    """Test signal generation with invalid data."""
    input_data_ml = {
        'symbol': 'BTC/USDT',
        'trend': 'unknown',
        'strength': Decimal('0.8')
    }
    
    with pytest.raises(ValidationError):
        generate_ml_signals(input_data_ml)
    
    input_data_ga = {
        'symbol': 'ETH/USDT',
        'strategy': 'unknown',
        'indicator_values': {'macd': Decimal('0.5'), 'rsi': Decimal('30')}
    }
    
    with pytest.raises(ValidationError):
        generate_ga_signals(input_data_ga) 