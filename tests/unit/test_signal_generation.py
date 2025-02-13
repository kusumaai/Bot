import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from src.signals.ml_signals import generate_ml_signals
from src.signals.ga_synergy import generate_ga_signals
from src.utils.error_handler import ValidationError


def test_generate_ml_signals_bullish():
    """Test ML signal generation for bullish trend."""
    input_data = {
        'symbol': 'BTC/USDT',
        'trend': 'bullish',
        'strength': Decimal('0.8')
    }
    
    signals = generate_ml_signals(input_data)
    assert signals['signal'] == 'buy'
    assert signals['confidence'] == Decimal('0.8')


def test_generate_ml_signals_bearish():
    """Test ML signal generation for bearish trend."""
    input_data = {
        'symbol': 'BTC/USDT',
        'trend': 'bearish',
        'strength': Decimal('0.7')
    }
    
    signals = generate_ml_signals(input_data)
    assert signals['signal'] == 'sell'
    assert signals['confidence'] == Decimal('0.7')


def test_generate_ml_signals_invalid_trend():
    """Test ML signal generation with invalid trend."""
    input_data = {
        'symbol': 'BTC/USDT',
        'trend': 'neutral',
        'strength': Decimal('0.5')
    }
    
    with pytest.raises(ValidationError, match="Invalid trend: neutral"):
        generate_ml_signals(input_data)


def test_generate_ga_signals_crossover():
    """Test GA signal generation for crossover strategy."""
    input_data = {
        'symbol': 'ETH/USDT',
        'strategy': 'crossover',
        'indicator_values': {'macd': Decimal('0.5'), 'rsi': Decimal('30')}
    }
    
    signals = generate_ga_signals(input_data)
    assert signals['signal'] == 'buy'
    assert signals['strategy'] == 'crossover'


def test_generate_ga_signals_platform_break():
    """Test GA signal generation for platform break strategy."""
    input_data = {
        'symbol': 'ETH/USDT',
        'strategy': 'platform_break',
        'indicator_values': {'volume': Decimal('1500')}
    }
    
    signals = generate_ga_signals(input_data)
    assert signals['signal'] == 'sell'
    assert signals['strategy'] == 'platform_break'


def test_generate_ga_signals_invalid_strategy():
    """Test GA signal generation with invalid strategy."""
    input_data = {
        'symbol': 'ETH/USDT',
        'strategy': 'unknown_strategy',
        'indicator_values': {'volume': Decimal('1500')}
    }
    
    with pytest.raises(ValidationError, match="Invalid strategy: unknown_strategy"):
        generate_ga_signals(input_data) 