#! /usr/bin/env python3
# src/database/sentiment_data.py
"""
Module: src.database.sentiment_data
Provides functions to retrieve sentiment and auxiliary data from the database using SQLAlchemy for robust access and security.
"""

import logging
from typing import Any, Dict, List

from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


def get_sentiment_data(db_url: str) -> List[Dict[str, Any]]:
    """
    Retrieve sentiment data from the database using SQLAlchemy.
    Assumes a table named 'sentiments' exists.
    Uses connection pooling and parameterized queries for security and performance.

    :param db_url: Database URL (example: 'sqlite:///data/trading.db')
    :return: List of dictionaries representing sentiment data
    """
    try:
        # Create engine with connection pooling and pre-ping for stale connections
        engine = create_engine(db_url, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        metadata = MetaData()
        # Reflect the sentiments table from the database
        sentiments = Table("sentiments", metadata, autoload_with=engine)

        stmt = select(sentiments)
        results = session.execute(stmt).fetchall()
        column_names = sentiments.columns.keys()
        session.close()
        data = [dict(zip(column_names, row)) for row in results]
        return data
    except Exception as e:
        logger.error(f"Error retrieving sentiment data: {e}", exc_info=True)
        return []
