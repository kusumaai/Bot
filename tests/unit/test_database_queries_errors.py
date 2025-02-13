import pytest
import asyncio
from src.database.queries import DatabaseQueries
from src.utils.error_handler import DatabaseError

@pytest.fixture
def fake_db_connection():
    # Fake connection that always fails
    class FakeDB:
        async def execute(self, query, params=None):
            raise Exception("DB failure")
    return FakeDB()

@pytest.fixture
def logger():
    import logging
    return logging.getLogger("test_db")

@pytest.fixture
def db_queries(fake_db_connection, logger):
    return DatabaseQueries(connection=fake_db_connection, logger=logger)

@pytest.mark.asyncio
async def test_db_store_trade_failure(db_queries):
    invalid_trade = {
        'id': 'trade_fail',
        'symbol': 'BTC/USDT'
        # Missing other required fields
    }
    with pytest.raises(DatabaseError):
        await db_queries.store_trade(invalid_trade) 