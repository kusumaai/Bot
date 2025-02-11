import pytest
from decimal import Decimal
from typing import Dict, Any
from risk.manager import RiskManager
from risk.position import Position
from risk.limits import RiskLimits

def test_position_size_calculation():
    config = {
        "max_position_size": "0.1",
        "max_positions": 3,
        "max_leverage": "2.0"
    }
    rm = RiskManager(config)
    
    # Test basic calculation
    signal = {"probability": "0.6"}
    market_data = {
        "current_price": "100.0",
        "volatility": "0.2"
    }
    result = rm.calculate_position_size(signal, market_data)
    assert isinstance(result.size, Decimal)
    assert result.size > 0
    
    # Test max position size limit
    signal = {"probability": "1.0"}  # Strong signal
    result = rm.calculate_position_size(signal, market_data)
    assert result.size <= Decimal("0.1")  # Should not exceed max 