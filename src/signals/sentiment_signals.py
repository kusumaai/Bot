#! /usr/bin/env python3
# src/signals/sentiment_signals.py
"""
Module: src.signals.sentiment_signals
Generates trading signals based on sentiment analysis.
Provides a robust implementation for production use.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from database.sentiment_data import get_sentiment_data

logger = logging.getLogger(__name__)


def generate_sentiment_signal(
    db_url: str, symbol: str, buy_threshold: float = 0.6, sell_threshold: float = 0.4
) -> Optional[Dict[str, Any]]:
    """
    Generate a trading signal based on sentiment analysis.

    This function retrieves sentiment data from the database, filters data for the given symbol,
    and calculates the average sentiment score. If the average sentiment is greater than or equal
    to the buy_threshold, a 'buy' signal is generated. If the average sentiment is less than or equal
    to the sell_threshold, a 'sell' signal is generated. Otherwise, no signal is returned.

    :param db_url: Database URL for retrieving sentiment data (e.g., 'sqlite:///data/trading.db')
    :param symbol: Trading symbol to filter sentiment data
    :param buy_threshold: Threshold above which to generate a 'buy' signal (default 0.6)
    :param sell_threshold: Threshold below which to generate a 'sell' signal (default 0.4)
    :return: A dictionary representing the trading signal, or None if no signal is generated
    """
    try:
        data = get_sentiment_data(db_url)
        if not data:
            logger.warning("No sentiment data available.")
            return None

        # Filter sentiment data for the given symbol
        symbol_data = [record for record in data if record.get("symbol") == symbol]
        if not symbol_data:
            logger.info(f"No sentiment records found for symbol '{symbol}'.")
            return None

        # Compute average sentiment score
        total_score = 0.0
        valid_count = 0
        for record in symbol_data:
            try:
                score = float(record.get("score", 0))
                total_score += score
                valid_count += 1
            except Exception as e:
                logger.error(
                    f"Error parsing sentiment score from record {record}: {e}",
                    exc_info=True,
                )
        if valid_count == 0:
            logger.error("No valid sentiment scores available.")
            return None
        avg_sentiment = total_score / valid_count
        logger.info(f"Average sentiment for {symbol}: {avg_sentiment:.3f}")

        signal: Optional[Dict[str, Any]] = None
        if avg_sentiment >= buy_threshold:
            signal = {
                "symbol": symbol,
                "action": "buy",
                "confidence": Decimal(str(avg_sentiment)),
                "source": "sentiment_analysis",
                "timestamp": datetime.utcnow().isoformat(),
            }
        elif avg_sentiment <= sell_threshold:
            signal = {
                "symbol": symbol,
                "action": "sell",
                "confidence": Decimal(str(1 - avg_sentiment)),
                "source": "sentiment_analysis",
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            logger.info(
                f"Sentiment for {symbol} is neutral (avg: {avg_sentiment:.3f}); no signal generated."
            )

        return signal

    except Exception as e:
        logger.error(f"Error generating sentiment signal: {e}", exc_info=True)
        return None
