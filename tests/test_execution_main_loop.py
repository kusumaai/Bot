from unittest import mock
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.execution.main_loop import main_loop

@pytest.fixture
def mock_ctx():
    mock = MagicMock()
    mock.logger = MagicMock()
    mock.market_data.get_signals = AsyncMock(return_value=[])
    mock.order_manager.place_order = AsyncMock()
    mock.running = False  # To stop the loop after one iteration
    return mock

@pytest.mark.asyncio
async def test_main_loop_no_signals(mock_ctx):
    await main_loop(mock_ctx)
    mock_ctx.market_data.get_signals.assert_called()
    mock_ctx.order_manager.place_order.assert_not_called()

@pytest.mark.asyncio
async def test_main_loop_with_signals(mock_ctx):
    mock_ctx.market_data.get_signals = AsyncMock(return_value=[
        {"symbol": "BTCUSD", "action": "buy", "price": "50000", "quantity": "1"}
    ])
    mock_ctx.order_manager.place_order = AsyncMock(return_value=None)
    await main_loop(mock_ctx)
    mock_ctx.market_data.get_signals.assert_called()
    mock_ctx.order_manager.place_order.assert_called_with({"symbol": "BTCUSD", "action": "buy", "price": "50000", "quantity": "1"}) 