#! /usr/bin/env python3
#tests/unit/test_symbols.py
"""
Module: tests.unit
Provides unit testing functionality for the symbols module.
""" 
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.execution.market_data import MarketData
from src.utils.error_handler import ValidationError


@pytest.fixture
def market_data_fixture():
    """Provide a MarketData instance."""
    return MarketData(
        db_queries=AsyncMock(),
        logger=MagicMock()
    )


def test_normalize_symbol(market_data_fixture):
    """Test normalization of symbols."""
    normalized = market_data_fixture.normalize_symbol('btc_usdt')
    assert normalized == 'BTC/USDT'
    
    normalized = market_data_fixture.normalize_symbol('eth-usdt')
    assert normalized == 'ETH/USDT'
    
    normalized = market_data_fixture.normalize_symbol('SOLUSDT')
    assert normalized == 'SOL/USDT'


def test_validate_symbol_format(market_data_fixture):
    """Test validation of symbol formats."""
    valid_symbol = 'BTC/USDT'
    assert market_data_fixture.validate_symbol_format(valid_symbol) is True
    
    invalid_symbol = 'BTCUSDT'
    with pytest.raises(ValidationError, match="Invalid symbol format"):
        market_data_fixture.validate_symbol_format(invalid_symbol)
    
    invalid_symbol = 'BTC-USD-T'
    with pytest.raises(ValidationError, match="Invalid symbol format"):
        market_data_fixture.validate_symbol_format(invalid_symbol) 