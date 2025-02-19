#! /usr/bin/env python3
# tests/unit/test_signal_generation.py
"""
Module: tests.unit
Provides unit testing functionality for the signal generation module.
"""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from signals.ga_synergy import generate_ga_signals
from signals.ml_signals import generate_ml_signals
from utils.error_handler import ValidationError


# ML Signals
@pytest.mark.asyncio
async def test_generate_ml_signals_bullish():
    """Test ML signal generation for bullish trend."""
    input_data = {"symbol": "BTC/USDT", "trend": "bullish", "strength": Decimal("0.8")}
    mock_ctx = MagicMock()
    signals = await generate_ml_signals(mock_ctx, input_data)
    assert signals["action"] == "buy"
    assert signals["strength"] == Decimal("0.8")


# ML Signals
@pytest.mark.asyncio
async def test_generate_ml_signals_bearish():
    """Test ML signal generation for bearish trend."""
    input_data = {"symbol": "BTC/USDT", "trend": "bearish", "strength": Decimal("0.7")}
    mock_ctx = MagicMock()
    signals = await generate_ml_signals(mock_ctx, input_data)
    assert signals["action"] == "sell"
    assert signals["strength"] == Decimal("0.7")


# Invalid Trend
@pytest.mark.asyncio
async def test_generate_ml_signals_invalid_trend():
    """Test ML signal generation with invalid trend."""
    input_data = {"symbol": "BTC/USDT", "trend": "neutral", "strength": Decimal("0.5")}
    mock_ctx = MagicMock()
    with pytest.raises(ValidationError, match="Invalid trend: neutral"):
        await generate_ml_signals(mock_ctx, input_data)


# GA Signals
def test_generate_ga_signals_crossover():
    """Test GA signal generation for crossover strategy."""
    input_data = {
        "symbol": "ETH/USDT",
        "strategy": "crossover",
        "indicator_values": {"macd": Decimal("0.5"), "rsi": Decimal("30")},
    }
    signals = generate_ga_signals(input_data, population=100)
    assert isinstance(signals, dict)
    assert signals["type"] == "ga"
    assert signals["strategy"] == "crossover"


# GA Signals
def test_generate_ga_signals_platform_break():
    """Test GA signal generation for platform break strategy."""
    input_data = {
        "symbol": "ETH/USDT",
        "strategy": "platform_break",
        "indicator_values": {"volume": Decimal("1500")},
    }
    signals = generate_ga_signals(input_data, population=100)
    assert isinstance(signals, dict)
    assert signals["type"] == "ga"
    assert signals["strategy"] == "platform_break"


# Invalid Strategy
def test_generate_ga_signals_invalid_strategy():
    """Test GA signal generation with invalid strategy."""
    input_data = {
        "symbol": "ETH/USDT",
        "strategy": "unknown_strategy",
        "indicator_values": {"volume": Decimal("1500")},
    }
    with pytest.raises(ValidationError, match="Invalid strategy: unknown_strategy"):
        generate_ga_signals(input_data, population=100)
