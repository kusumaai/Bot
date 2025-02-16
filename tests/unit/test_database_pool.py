import asyncio
import sqlite3
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from database.database import DatabaseConnection, DatabaseError
from utils.numeric_handler import NumericHandler


@pytest.fixture
async def db_connection():
    """Create a test database connection with in-memory SQLite."""
    connection = DatabaseConnection(":memory:", NumericHandler())
    await connection.initialize()
    yield connection
    await connection.close()


@pytest.mark.asyncio
async def test_connection_pool_initialization(db_connection):
    """Test that the connection pool initializes with correct number of connections."""
    assert db_connection.initialized
    assert len(db_connection._connection_pool) >= db_connection.min_connections
    assert len(db_connection._connection_pool) <= db_connection.max_connections


@pytest.mark.asyncio
async def test_get_connection(db_connection):
    """Test getting a connection from the pool."""
    conn = await db_connection.get_connection()
    assert conn is not None
    async with conn.cursor() as cursor:
        await cursor.execute("SELECT 1")
        result = await cursor.fetchone()
        assert result[0] == 1
    await db_connection.return_connection(conn)


@pytest.mark.asyncio
async def test_connection_pool_exhaustion():
    """Test behavior when connection pool is exhausted."""
    connection = DatabaseConnection(":memory:", NumericHandler())
    connection.max_connections = 2
    await connection.initialize()

    try:
        # Get all available connections
        conn1 = await connection.get_connection()
        conn2 = await connection.get_connection()

        # This should raise an error as pool is exhausted
        with pytest.raises(DatabaseError):
            await connection.get_connection()

        # Return connections and verify we can get new ones
        await connection.return_connection(conn1)
        await connection.return_connection(conn2)

        # Should be able to get a connection now
        conn = await connection.get_connection()
        assert conn is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_connection_retry_logic():
    """Test connection retry logic with simulated failures."""
    with patch("aiosqlite.connect") as mock_connect:
        # Make first two attempts fail, third succeed
        mock_conn = AsyncMock()
        mock_connect.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            mock_conn,
        ]

        connection = DatabaseConnection(":memory:", NumericHandler())
        connection.retry_delay = 0.1  # Speed up test
        await connection.initialize()

        assert connection.initialized
        assert mock_connect.call_count >= 3
        await connection.close()


@pytest.mark.asyncio
async def test_transaction_management(db_connection):
    """Test transaction management with commit and rollback."""
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """
            )
            await cursor.execute("INSERT INTO test (value) VALUES (?)", ("test",))

    # Verify data was committed
    async with db_connection.get_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT value FROM test")
            result = await cursor.fetchone()
            assert result[0] == "test"

    # Test rollback
    try:
        async with db_connection.transaction() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("INSERT INTO test (value) VALUES (?)", ("test2",))
                raise Exception("Forced rollback")
    except Exception:
        pass

    # Verify data was rolled back
    async with db_connection.get_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM test")
            result = await cursor.fetchone()
            assert result[0] == 1  # Only the first insert should remain


@pytest.mark.asyncio
async def test_execute_with_retry(db_connection):
    """Test query execution with retry logic."""
    # Create test table
    async with db_connection.get_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE retry_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """
            )

    # Test successful execution
    result = await db_connection.execute_with_retry(
        "INSERT INTO retry_test (value) VALUES (?) RETURNING id", ("test_value",)
    )
    assert result is not None

    # Test retry on database locked error
    with patch("aiosqlite.Connection.execute") as mock_execute:
        mock_execute.side_effect = [
            Exception("database is locked"),
            Exception("database is locked"),
            AsyncMock(return_value=AsyncMock(fetchall=AsyncMock(return_value=[(1,)]))),
        ]

        result = await db_connection.execute_with_retry(
            "SELECT id FROM retry_test WHERE value = ?", ("test_value",)
        )
        assert result == [(1,)]
        assert mock_execute.call_count == 3


@pytest.mark.asyncio
async def test_execute_many(db_connection):
    """Test batch execution of queries."""
    # Create test table
    async with db_connection.get_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE batch_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """
            )

    # Prepare batch data
    batch_data = [("value1",), ("value2",), ("value3",)]

    # Execute batch insert
    await db_connection.execute_many(
        "INSERT INTO batch_test (value) VALUES (?)", batch_data, batch_size=2
    )

    # Verify results
    async with db_connection.get_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM batch_test")
            count = await cursor.fetchone()
            assert count[0] == 3

            await cursor.execute("SELECT value FROM batch_test ORDER BY id")
            results = await cursor.fetchall()
            assert [r[0] for r in results] == ["value1", "value2", "value3"]


@pytest.mark.asyncio
async def test_nested_transactions(db_connection):
    """Test nested transaction handling with savepoints."""
    # Create test table
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE nested_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
                """
            )

    try:
        # Outer transaction
        async with db_connection.transaction() as conn1:
            async with conn1.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO nested_test (value) VALUES (?)",
                    ("outer",),
                )

                # Nested transaction that succeeds
                async with db_connection.transaction() as conn2:
                    async with conn2.cursor() as cursor:
                        await cursor.execute(
                            "INSERT INTO nested_test (value) VALUES (?)",
                            ("nested_success",),
                        )

                # Nested transaction that fails
                try:
                    async with db_connection.transaction() as conn3:
                        async with conn3.cursor() as cursor:
                            await cursor.execute(
                                "INSERT INTO nested_test (value) VALUES (?)",
                                ("nested_fail",),
                            )
                            raise ValueError("Forced nested transaction failure")
                except ValueError:
                    pass  # Expected failure

        # Verify results
        async with db_connection.transaction() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT value FROM nested_test ORDER BY id")
                results = await cursor.fetchall()
                assert len(results) == 2
                assert results[0][0] == "outer"
                assert results[1][0] == "nested_success"
                assert not any(r[0] == "nested_fail" for r in results)

    except Exception as e:
        assert False, f"Unexpected error: {e}"


@pytest.mark.asyncio
async def test_transaction_timeout(db_connection):
    """Test transaction timeout handling."""
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE timeout_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
                """
            )

    # Set a very short timeout to force timeout condition
    with pytest.raises(DatabaseError, match="Query timeout"):
        await db_connection.execute_with_retry(
            "INSERT INTO timeout_test (value) VALUES (?)",
            ("test",),
            timeout=0.001,
        )


@pytest.mark.asyncio
async def test_transaction_deadlock_recovery(db_connection):
    """Test recovery from transaction deadlocks."""
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE deadlock_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
                """
            )

    # Simulate deadlock by raising sqlite3.Error with deadlock message
    with patch("sqlite3.Connection.execute") as mock_execute:
        mock_execute.side_effect = [
            sqlite3.Error("database is deadlocked"),
            sqlite3.Error("database is deadlocked"),
            None,  # Success on third try
        ]

        # Should succeed after retries
        await db_connection.execute_with_retry(
            "INSERT INTO deadlock_test (value) VALUES (?)",
            ("test",),
        )

        assert mock_execute.call_count == 3


@pytest.mark.asyncio
async def test_batch_execution(db_connection):
    """Test batch query execution with atomic and non-atomic modes."""
    # Create test table
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE batch_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
                """
            )

    # Test atomic batch execution
    queries = [
        ("INSERT INTO batch_test (value) VALUES (?)", ("value1",)),
        ("INSERT INTO batch_test (value) VALUES (?)", ("value2",)),
        ("INSERT INTO batch_test (value) VALUES (?)", ("value3",)),
    ]

    # Successful atomic batch
    success = await db_connection.execute_batch(queries, atomic=True)
    assert success

    # Verify results
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM batch_test")
            count = await cursor.fetchone()
            assert count[0] == 3

    # Test atomic batch with failure
    bad_queries = [
        ("INSERT INTO batch_test (value) VALUES (?)", ("value4",)),
        ("INSERT INTO bad_table (value) VALUES (?)", ("value5",)),  # This will fail
    ]

    with pytest.raises(DatabaseError):
        await db_connection.execute_batch(bad_queries, atomic=True)

    # Verify no changes were made due to atomic failure
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM batch_test")
            count = await cursor.fetchone()
            assert count[0] == 3  # Still 3 from previous successful batch

    # Test non-atomic batch execution
    partial_queries = [
        ("INSERT INTO batch_test (value) VALUES (?)", ("value6",)),
        ("INSERT INTO bad_table (value) VALUES (?)", ("value7",)),  # This will fail
        ("INSERT INTO batch_test (value) VALUES (?)", ("value8",)),
    ]

    success = await db_connection.execute_batch(partial_queries, atomic=False)
    assert not success  # Overall failure due to middle query

    # Verify partial execution
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT value FROM batch_test WHERE value IN (?, ?)",
                ("value6", "value8"),
            )
            results = await cursor.fetchall()
            assert len(results) == 2  # value6 and value8 should be inserted


@pytest.mark.asyncio
async def test_transaction_cleanup(db_connection):
    """Test proper cleanup of transactions during shutdown."""
    # Start some transactions
    async with db_connection.transaction() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                CREATE TABLE cleanup_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
                """
            )

    # Simulate a failed transaction
    try:
        async with db_connection.transaction() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO cleanup_test (value) VALUES (?)",
                    ("test",),
                )
                raise ValueError("Forced failure")
    except ValueError:
        pass

    # Close the database connection
    await db_connection.close()

    # Verify cleanup
    assert not db_connection._active_transactions
    assert not db_connection._transaction_states
    assert not db_connection._savepoint_counter
    assert not db_connection.initialized
