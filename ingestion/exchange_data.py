#!/usr/bin/env python3
"""
Module: ingestion/exchange_data.py
Handles exchange data ingestion with proper error handling and rate limiting
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from collections import deque
import logging

from utils import logger
from utils.error_handler import handle_error, ExchangeError
from exchanges.exchange_manager import ExchangeManager
from data.candles import CandleProcessor
from indicators.quality_monitor import DataQualityMonitor
from database.queries import DatabaseQueries

# Global cache for exchange instances with timeout management
class ExchangeCache:
    def __init__(self):
        self.instances: Dict[str, "ExchangeInstance"] = {}
        self.last_access: Dict[str, float] = {}
        self.cache_timeout = 3600  # 1 hour default

    def get(self, eid: str) -> Optional["ExchangeInstance"]:
        if eid not in self.instances:
            return None
        
        last_access = self.last_access.get(eid, 0)
        if time.time() - last_access > self.cache_timeout:
            del self.instances[eid]
            return None
            
        self.last_access[eid] = time.time()
        return self.instances[eid]

    def set(self, eid: str, instance: "ExchangeInstance") -> None:
        self.instances[eid] = instance
        self.last_access[eid] = time.time()

_exchange_cache = ExchangeCache()

async def get_exchange_instance(eid: str, ctx: Any) -> ExchangeManager:
    """Get or create an exchange instance with proper caching and validation"""
    cached = _exchange_cache.get(eid)
    if cached:
        return cached

    try:
        exchange_instance = ExchangeManager(ctx)
        await exchange_instance.initialize()
        _exchange_cache.set(eid, exchange_instance)
        return exchange_instance
    except ExchangeError as e:
        ctx.logger.error(f"Failed to get exchange instance for {eid}: {e}")
        raise e
    except Exception as e:
        ctx.logger.error(f"Unexpected error while getting exchange instance for {eid}: {e}")
        raise ExchangeError(f"Failed to get exchange instance for {eid}: {e}") from e

class ExchangeInstance:
    """Placeholder for ExchangeInstance class if needed"""
    pass  # Implement if required

class IngestionService:
    def __init__(self, ctx: Any, batch_size: int = 100):
        self.ctx = ctx
        self.logger = ctx.logger
        self.batch_size = batch_size
        self.processor = CandleProcessor(DatabaseQueries(ctx.config.get("database", {}).get("path", "data/trading.db")), logger=self.logger)
        self.quality_monitor = DataQualityMonitor(logger)
        
        self._running = False
        self._last_fetch: Dict[str, datetime] = {}
        
    async def start_ingestion(
        self,
        symbols: List[str],
        timeframe: str,
        interval: int = 60
    ) -> None:
        self._running = True
        while self._running:
            try:
                tasks = [
                    self._ingest_symbol_data(symbol, timeframe)
                    for symbol in symbols
                ]
                await asyncio.gather(*tasks)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Ingestion error: {str(e)}")
                await asyncio.sleep(5)  # Brief pause on error
    
    def stop_ingestion(self) -> None:
        self._running = False
    
    async def _ingest_symbol_data(
        self,
        symbol: str,
        timeframe: str
    ) -> None:
        try:
            now = datetime.utcnow()
            last_fetch = self._last_fetch.get(symbol)
            
            if last_fetch and (now - last_fetch) < timedelta(minutes=1):
                return
                
            candles = await self.ctx.exchange_interface.fetch_candles(
                symbol=symbol,
                timeframe=timeframe,
                limit=self.batch_size
            )
            
            if not candles:
                return
                
            df = await self.processor.process_candles(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles
            )
            
            quality_result = self.quality_monitor.check_data_quality(symbol, df)
            if quality_result['issues']:
                self.logger.warning(
                    f"Data quality issues for {symbol}: {quality_result['issues']}"
                )
            
            self._last_fetch[symbol] = now
            
        except ExchangeError as e:
            self.logger.error(f"Exchange error for {symbol}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Ingestion error for {symbol}: {str(e)}")
