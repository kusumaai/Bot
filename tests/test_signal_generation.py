from decimal import Decimal
import pytest
from src.signals.ga_synergy import generate_ga_signals, GASignal
from src.utils.exceptions import InvalidOrderError
from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_generate_ga_signals_valid():
    ctx = MagicMock()  # Assuming 'ctx' is needed
    population = 50  # Example population size
    signals = await generate_ga_signals(ctx, population=population)
    
    assert isinstance(signals, list)
    assert len(signals) == population
    # Add more specific assertions based on expected signal structure

@pytest.mark.asyncio
async def test_generate_ga_signals_invalid_action():
    ctx = MagicMock()
    population = 50
    # Modify ctx or inputs to create invalid action
    with pytest.raises(ValueError):
        await generate_ga_signals(ctx, population=population) 