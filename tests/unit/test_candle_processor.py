import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from data.candles import CandleProcessor
from database.queries import DatabaseQueries
from utils.error_handler import ValidationError
from utils.numeric_handler import NumericHandler


@pytest.fixture
def sample_candles() -> List[List]:
    """Create sample candle data for testing."""
    now = datetime.now()
    return [
        # timestamp, open, high, low, close, volume
        [int(now.timestamp() * 1000), "100.0", "105.0", "98.0", "103.0", "1000.0"],
        [
            int((now + timedelta(minutes=15)).timestamp() * 1000),
            "103.0",
            "108.0",
            "102.0",
            "107.0",
            "1500.0",
        ],
        [
            int((now + timedelta(minutes=30)).timestamp() * 1000),
            "107.0",
            "110.0",
            "105.0",
            "106.0",
            "2000.0",
        ],
    ]


@pytest.fixture
def mock_db_queries():
    """Create mock database queries."""
    return MagicMock(spec=DatabaseQueries)


@pytest.fixture
def candle_processor(mock_db_queries):
    """Create candle processor instance."""
    logger = logging.getLogger("test_logger")
    return CandleProcessor(mock_db_queries, logger)


@pytest.mark.asyncio
async def test_process_candles_success(candle_processor, sample_candles):
    """Test successful processing of valid candle data."""
    df = await candle_processor.process_candles("BTC/USDT", "15m", sample_candles)

    assert not df.empty
    assert len(df) == 3
    assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
    assert isinstance(df["open"][0], Decimal)
    assert isinstance(df["volume"][0], Decimal)

    # Verify calculated metrics
    assert "returns" in df.columns
    assert "tr" in df.columns
    assert "atr_14" in df.columns
    assert "volume_ma" in df.columns
    assert "volume_std" in df.columns


@pytest.mark.asyncio
async def test_process_candles_empty_data(candle_processor):
    """Test handling of empty candle data."""
    df = await candle_processor.process_candles("BTC/USDT", "15m", [])
    assert df.empty


@pytest.mark.asyncio
async def test_safe_decimal_convert(candle_processor):
    """Test safe conversion to Decimal."""
    assert candle_processor._safe_decimal_convert("100.0") == Decimal("100.0")
    assert candle_processor._safe_decimal_convert(100.0) == Decimal("100.0")
    assert candle_processor._safe_decimal_convert(None) == Decimal("0")
    assert candle_processor._safe_decimal_convert("invalid") == Decimal("0")


@pytest.mark.asyncio
async def test_calculate_metrics(candle_processor, sample_candles):
    """Test calculation of additional metrics."""
    df = pd.DataFrame(
        sample_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.apply(pd.to_numeric)

    result_df = await candle_processor._calculate_metrics(df)

    assert "returns" in result_df.columns
    assert "tr" in result_df.columns
    assert "atr_14" in result_df.columns
    assert "volume_ma" in result_df.columns
    assert "volume_std" in result_df.columns

    # Verify calculations are not null
    assert not result_df["returns"].isna().all()
    assert not result_df["tr"].isna().all()
    assert not result_df["atr_14"].isna().all()
    assert not result_df["volume_ma"].isna().all()
    assert not result_df["volume_std"].isna().all()


@pytest.mark.asyncio
async def test_processing_stats(candle_processor, sample_candles):
    """Test processing statistics tracking."""
    await candle_processor.process_candles("BTC/USDT", "15m", sample_candles)

    stats = candle_processor.get_processing_stats()
    assert stats["total_processed"] > 0
    assert stats["total_errors"] == 0
    assert stats["last_success"] is not None
    assert stats["error_rate"] == 0.0
