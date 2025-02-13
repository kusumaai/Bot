import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from src.execution.main_loop import MainLoop
from src.execution.exchange_interface import ExchangeInterface
from src.risk.manager import RiskManager
from src.database.queries import DatabaseQueries
from src.utils.logger import setup_logging

@pytest.fixture
def fake_exchange_interface():
    fake = MagicMock(spec=ExchangeInterface)
    fake.fetch_market_data = AsyncMock(return_value={'BTC/USDT': {'price': Decimal('50000')}})
    fake.execute_trade = AsyncMock(return_value={'success': True, 'order_id': 'order456'})
    return fake

@pytest.fixture
def fake_risk_manager():
    fake = MagicMock(spec=RiskManager)
    fake.validate_trade = AsyncMock(return_value=(True, None))
    return fake

@pytest.fixture
def fake_db_queries():
    fake = MagicMock(spec=DatabaseQueries)
    fake.store_trade = AsyncMock(return_value=True)
    return fake

@pytest.fixture
def logger():
    return setup_logging("integrationTest", "DEBUG")

@pytest.fixture
def main_loop(fake_exchange_interface, fake_risk_manager, fake_db_queries, logger):
    return MainLoop(fake_exchange_interface, fake_risk_manager, fake_db_queries, logger)

@pytest.mark.asyncio
async def test_full_trade_flow(main_loop):
    # Here, you could trigger a single iteration of the main loop if a method exists (e.g., run_once)
    result = await main_loop.run_once()  # You may need to expose or simulate one iteration.
    assert result is True 