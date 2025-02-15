#!/usr/bin/env python3
"""
Module: accounting/accounting.py
"""
import asyncio
import time
from decimal import Decimal

# import the necessary libraries
from typing import Any, Dict, List, Optional

from database.database import DatabaseConnection, execute_sql, execute_sql_one
from utils.error_handler import handle_error, handle_error_async
from utils.exceptions import DatabaseError


# validate the account for the trading bot
def validate_account(signal: Dict[str, Any], ctx: Any) -> bool:
    """Validate if account has sufficient free balance."""
    try:
        min_free_balance = ctx.config.get("min_free_balance_threshold", 0.001)
        with DatabaseConnection(ctx.db_pool) as conn:
            row = execute_sql_one(
                conn,
                "SELECT balance, used_balance FROM account WHERE exchange = ?",
                [signal["exchange"]],
            )
            if not row:
                ctx.logger.warning(
                    f"No account found for exchange {signal['exchange']}"
                )
                return False

            free_balance = Decimal(str(row["balance"])) - Decimal(
                str(row["used_balance"])
            )
            return free_balance >= Decimal(str(min_free_balance))

    except Exception as e:
        handle_error(e, context="Accounting.validate_account", logger=ctx.logger)
        return False


# get the free balance for the trading bot
def get_free_balance(exchange: str, ctx: Any) -> Decimal:
    """Get available balance for trading."""
    try:
        with DatabaseConnection(ctx.db_pool) as conn:
            is_paper = ctx.config.get("paper_mode", False)
            sql = """
                SELECT 
                    CASE 
                        WHEN ? THEN paper_balance - paper_used_balance
                        ELSE balance - used_balance 
                    END as free_balance
                FROM account 
                WHERE exchange = ?
            """
            row = execute_sql_one(conn, sql, [is_paper, exchange])
            if not row:
                ctx.logger.warning(f"No account found for exchange {exchange}")
                return Decimal("0")

            return Decimal(str(row["free_balance"]))

    except Exception as e:
        handle_error(e, context="Accounting.get_free_balance", logger=ctx.logger)
        return Decimal("0")


# record a new trade for the trading bot
def record_new_trade(
    order: Dict[str, Any],
    signal: Dict[str, Any],
    ev: float,
    kelly_frac: float,
    position_size: float,
    ctx: Any,
    trade_source: str = "real",
) -> bool:
    """Record a new trade and update account balance."""
    try:
        with DatabaseConnection(ctx.db_pool) as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                # Insert trade record
                sql_ins = """
                    INSERT INTO trades (
                        id, symbol, timeframe, trade_source, direction,
                        entry_price, sl, tp, entry_time, exchange, position_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)
                """
                params_ins = [
                    order["id"],
                    signal["symbol"],
                    ctx.config["timeframe"],
                    trade_source,
                    signal["direction"],
                    signal["entry_price"],
                    signal.get("sl", 0),
                    signal.get("tp", 0),
                    signal["exchange"],
                    position_size,
                ]
                execute_sql(conn, sql_ins, params_ins)

                # Update account balance based on trade source
                if trade_source == "real":
                    upd_sql = """
                        UPDATE account 
                        SET used_balance = used_balance + ? 
                        WHERE exchange = ? 
                        AND used_balance + ? <= balance
                    """
                    r = conn.cursor().execute(
                        upd_sql, [position_size, signal["exchange"], position_size]
                    )
                    if r.rowcount < 1:
                        raise ValueError("Insufficient free balance for this trade.")
                elif trade_source == "paper":
                    upd_sql = """
                        UPDATE account 
                        SET paper_used_balance = paper_used_balance + ? 
                        WHERE exchange = ? 
                        AND paper_used_balance + ? <= paper_balance
                    """
                    r = conn.cursor().execute(
                        upd_sql, [position_size, signal["exchange"], position_size]
                    )
                    if r.rowcount < 1:
                        raise ValueError(
                            "Insufficient free paper balance for this trade."
                        )

                conn.commit()
                return True

            except Exception as e:
                conn.rollback()
                raise e

    except Exception as e:
        handle_error(e, context="Accounting.record_new_trade", logger=ctx.logger)
        return False


# update the trade result for the trading bot
def update_trade_result(trade_id: str, net_pnl: float, ctx: Any) -> bool:
    """Update trade result and release used balance."""
    try:
        # Implementation to update trade result in the database
        # Example:
        trade = {
            "id": trade_id,
            "net_pnl": net_pnl,
            "timestamp": int(time.time() * 1000),
        }
        db_queries = ctx.db.queries  # Assuming ctx has db.queries attribute
        asyncio.create_task(db_queries.insert_trade(trade))
        return True
    except DatabaseError as e:
        asyncio.create_task(
            handle_error_async(e, "Accounting.update_trade_result", ctx.logger)
        )
        return False
    except Exception as e:
        asyncio.create_task(
            handle_error_async(e, "Accounting.update_trade_result", ctx.logger)
        )
        return False


# update the trade stop for the trading bot
def update_trade_stop(trade_id: str, new_sl: float, ctx: Any) -> None:
    """Update the stop loss for a trade."""
    try:
        with DatabaseConnection(ctx.db_pool) as conn:
            sql = "UPDATE trades SET sl = ? WHERE id = ?"
            execute_sql(conn, sql, [new_sl, trade_id])
    except Exception as e:
        handle_error(e, context="Accounting.update_trade_stop", logger=ctx.logger)


# fetch the open trades for the trading bot
def fetch_open_trades(ctx: Any) -> List[Dict[str, Any]]:
    """Fetch all open trades from the database."""
    try:
        with DatabaseConnection(ctx.db_pool) as conn:
            sql = (
                "SELECT * FROM trades "
                "WHERE close_reason IS NULL OR close_reason != 'closed' "
                "ORDER BY entry_time ASC"
            )
            rows = execute_sql(conn, sql, [])
            return rows
    except Exception as e:
        handle_error(e, context="Accounting.fetch_open_trades", logger=ctx.logger)
        return []


# update the daily performance for the trading bot
def update_daily_performance(ctx: Any) -> None:
    """Update daily performance statistics."""
    try:
        with DatabaseConnection(ctx.db_pool) as conn:
            sql = (
                "INSERT OR REPLACE INTO bot_performance ("
                "   day, real_trades_closed, paper_trades_closed, real_pnl, paper_pnl"
                ") VALUES ("
                "   date('now'),"
                "   (SELECT COUNT(*) FROM trades "
                "    WHERE close_reason='closed' AND trade_source='real' "
                "      AND date(entry_time)=date('now')),"
                "   (SELECT COUNT(*) FROM trades "
                "    WHERE close_reason='closed' AND trade_source='paper' "
                "      AND date(entry_time)=date('now')),"
                "   (SELECT COALESCE(SUM(result),0) FROM trades "
                "    WHERE close_reason='closed' AND trade_source='real' "
                "      AND date(entry_time)=date('now')),"
                "   (SELECT COALESCE(SUM(result),0) FROM trades "
                "    WHERE close_reason='closed' AND trade_source='paper' "
                "      AND date(entry_time)=date('now'))"
                ")"
            )
            execute_sql(conn, sql, [])
    except Exception as e:
        handle_error(
            e, context="Accounting.update_daily_performance", logger=ctx.logger
        )


# log the performance summary for the trading bot
def log_performance_summary(ctx: Any) -> None:
    """Log daily performance summary with better formatting"""
    try:
        with DatabaseConnection(ctx.db_pool) as conn:
            sql = "SELECT * FROM bot_performance ORDER BY day DESC LIMIT 1"
            summary = execute_sql_one(conn, sql, [])
            if summary:
                perf_str = (
                    f"Daily Performance - "
                    f"Real trades: {summary['real_trades_closed']}, "
                    f"Paper trades: {summary['paper_trades_closed']}, "
                    f"Real PnL: {summary['real_pnl']:.2f}, "
                    f"Paper PnL: {summary['paper_pnl']:.2f}"
                )
                ctx.logger.info(perf_str)
            else:
                ctx.logger.info("No performance data available yet")
    except Exception as e:
        handle_error(e, context="Accounting.log_performance_summary", logger=ctx.logger)
