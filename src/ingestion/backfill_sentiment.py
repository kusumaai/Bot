#! /usr/bin/env python3
#src/ingestion/backfill_sentiment.py
"""
Module: src.ingestion
Provides backfill sentiment management.
"""
import asyncio
import aiohttp
import json
import datetime
from typing import List, Dict, Any

from utils.error_handler import handle_error
from database.database import DBConnection
#fetch sentiment data from alternative.me (or a custom endpoint) asynchronously
async def fetch_sentiment_data(ctx: Any) -> List[Dict[str, Any]]:
    """
    Fetch sentiment data from Alternative.me (or a custom endpoint) asynchronously.
    
    Returns a list of dict records with:
      - date (YYYY-MM-DD)
      - fng_sentiment (float)
      - btc_dominance (float)
      - usdt_dominance (float)
      - usdc_dominance (float)
    """
    sentiment_cfg = ctx.config.get("sentiment", {})
    api_url   = sentiment_cfg.get("api_url", "https://api.alternative.me/fng/?limit=30&format=json")
    timeout_s = sentiment_cfg.get("api_timeout", 10)

    # Default fallback dominances if not configured
    btc_dom  = sentiment_cfg.get("btc_dominance", 45.0)
    usdt_dom = sentiment_cfg.get("usdt_dominance", 30.0)
    usdc_dom = sentiment_cfg.get("usdc_dominance", 25.0)

    timeout  = aiohttp.ClientTimeout(total=timeout_s)
    records  = []

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(api_url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error: {response.status}")
                response_text = await response.text()
                data = json.loads(response_text)

                if "data" not in data:
                    raise ValueError("API response missing 'data' field.")

                for item in data["data"]:
                    try:
                        ts    = int(item["timestamp"])
                        dt_str = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                        fng   = float(item["value"])
                        record = {
                            "date":           dt_str,
                            "fng_sentiment":  fng,
                            "btc_dominance":  btc_dom,
                            "usdt_dominance": usdt_dom,
                            "usdc_dominance": usdc_dom
                        }
                        records.append(record)
                    except Exception as inner_e:
                        handle_error(
                            inner_e,
                            context="fetch_sentiment_data: processing item",
                            logger=ctx.logger
                        )
    except Exception as e:
        handle_error(e, context="Ingestion.fetch_sentiment_data", logger=ctx.logger)

    return records

async def backfill_sentiment(ctx: Any) -> None:
    """
    Asynchronously fetch the sentiment data and insert it into 'sentiment_features'.
    If configured, retries the operation upon failure up to 'max_retries' times.
    """
    max_retries = ctx.config.get("sentiment", {}).get("max_retries", 1)
    attempt     = 0

    while attempt < max_retries:
        attempt += 1
        try:
            records = await fetch_sentiment_data(ctx)
            if not records:
                ctx.logger.warning("No sentiment records fetched from API.")
                if attempt < max_retries:
                    ctx.logger.info(f"Retrying sentiment fetch (attempt {attempt}/{max_retries})...")
                else:
                    ctx.logger.error("Sentiment fetch attempts exhausted.")
                continue  # Retry or exit loop

            inserted_count = 0
            for record in records:
                try:
                    with DBConnection(ctx.db_pool) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO sentiment_features
                            (date, fng_sentiment, btc_dominance, usdt_dominance, usdc_dominance)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                record["date"],
                                record["fng_sentiment"],
                                record["btc_dominance"],
                                record["usdt_dominance"],
                                record["usdc_dominance"]
                            )
                        )
                        conn.commit()
                        inserted_count += 1
                except Exception as db_e:
                    handle_error(db_e, context="backfill_sentiment DB insertion", logger=ctx.logger)

            ctx.logger.info(
                f"Sentiment backfill completed. Inserted/updated {inserted_count} records."
            )
            return  # Success, no further retries needed

        except Exception as e:
            handle_error(e, context="backfill_sentiment", logger=ctx.logger)
            if attempt < max_retries:
                ctx.logger.info(
                    f"Retrying sentiment backfill (attempt {attempt}/{max_retries})..."
                )
            else:
                ctx.logger.error("All backfill attempts failed; giving up.")
    # If while loop exits, all retries have failed.
