#! /usr/bin/env python3
#src/database/init_db.py    
"""
Database initialization script
Creates a fresh database with schema matching the codebase
"""
import sqlite3
import logging
import os
from pathlib import Path

def init_database(db_path: str) -> bool:
    """Initialize a fresh database with base schema"""
    logger = logging.getLogger(__name__)
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript("""
                BEGIN TRANSACTION;
                
                CREATE TABLE IF NOT EXISTS "account" (
                    "id" INTEGER,
                    "exchange" TEXT NOT NULL,
                    "balance" REAL NOT NULL,
                    "used_balance" REAL NOT NULL,
                    PRIMARY KEY("id")
                );
                
                CREATE TABLE IF NOT EXISTS "account_balance" (
                    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                    "balance" DECIMAL NOT NULL,
                    "timestamp" INTEGER NOT NULL,
                    "currency" TEXT NOT NULL DEFAULT 'USDT'
                );
                
                CREATE TABLE IF NOT EXISTS "bot_performance" (
                    "day" TEXT,
                    "real_trades_closed" INTEGER,
                    "paper_trades_closed" INTEGER,
                    "real_pnl" REAL,
                    "paper_pnl" REAL,
                    PRIMARY KEY("day")
                );
                
                CREATE TABLE IF NOT EXISTS "candles" (
                    "symbol" TEXT NOT NULL,
                    "timeframe" TEXT NOT NULL,
                    "timestamp" INTEGER NOT NULL,
                    "open" REAL NOT NULL,
                    "high" REAL NOT NULL,
                    "low" REAL NOT NULL,
                    "close" REAL NOT NULL,
                    "volume" REAL NOT NULL,
                    "datetime" TEXT NOT NULL,
                    "atr_14" REAL,
                    "exchange" TEXT,
                    PRIMARY KEY("symbol", "timeframe", "timestamp")
                );
                
                CREATE TABLE IF NOT EXISTS "ga_rules" (
                    "id" TEXT,
                    "chromosome_json" TEXT,
                    "fitness" REAL,
                    "date_created" TEXT,
                    PRIMARY KEY("id")
                );
                
                CREATE TABLE IF NOT EXISTS "sentiment_features" (
                    "date" TEXT,
                    "fng_sentiment" REAL,
                    "btc_dominance" REAL,
                    "usdt_dominance" REAL,
                    "usdc_dominance" REAL,
                    PRIMARY KEY("date")
                );
                
                CREATE TABLE IF NOT EXISTS "supported_pairs" (
                    "exchange" TEXT NOT NULL,
                    "symbol" TEXT NOT NULL,
                    "supported" INTEGER NOT NULL DEFAULT 1,
                    "last_checked" DATETIME NOT NULL,
                    PRIMARY KEY("exchange","symbol")
                );
                
                CREATE TABLE IF NOT EXISTS "trades" (
                    "id" TEXT,
                    "symbol" TEXT NOT NULL,
                    "timeframe" TEXT,
                    "trade_source" TEXT,
                    "direction" TEXT,
                    "entry_price" REAL,
                    "sl" REAL,
                    "tp" REAL,
                    "entry_time" TEXT,
                    "close_time" TEXT,
                    "result" REAL,
                    "close_reason" TEXT,
                    "exchange" TEXT,
                    "position_size" REAL,
                    PRIMARY KEY("id")
                );
                
                CREATE TABLE IF NOT EXISTS "positions" (
                    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                    "symbol" TEXT NOT NULL,
                    "side" TEXT NOT NULL,
                    "size" DECIMAL NOT NULL,
                    "entry_price" DECIMAL NOT NULL,
                    "status" TEXT NOT NULL,
                    "timestamp" INTEGER NOT NULL,
                    "current_price" DECIMAL,
                    "unrealized_pnl" DECIMAL,
                    "last_update" INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS "idx_candles_lookup" ON "candles"("symbol", "timeframe", "timestamp");
                CREATE INDEX IF NOT EXISTS "idx_trades_symbol" ON "trades"("symbol");
                CREATE INDEX IF NOT EXISTS "idx_trades_entry" ON "trades"("entry_time");
                CREATE INDEX IF NOT EXISTS "idx_positions_status" ON "positions"("status");
                CREATE INDEX IF NOT EXISTS "idx_account_balance_time" ON "account_balance"("timestamp");
                
                COMMIT;
            """)
            
            logger.info("Database initialized successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Get paths
    current_dir = Path(__file__).parent
    db_path = current_dir.parent / "data" / "candles.db"
    
    # Ensure data directory exists
    db_path.parent.mkdir(exist_ok=True)
    
    # Initialize fresh database
    if init_database(str(db_path)):
        logger.info(f"Database created at {db_path}")
        exit(0)
    else:
        logger.error("Failed to create database")
        exit(1) 