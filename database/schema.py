import logging


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS account_balance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance DECIMAL NOT NULL,
    timestamp INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size DECIMAL NOT NULL,
    entry_price DECIMAL NOT NULL,
    status TEXT NOT NULL,
    timestamp INTEGER NOT NULL
);
"""

async def initialize_schema(db_connection) -> bool:
    """Initialize database schema"""
    try:
        async with db_connection.get_connection() as conn:
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            return True
    except Exception as e:
        logging.error(f"Failed to initialize schema: {e}")
        return False
