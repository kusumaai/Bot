#!/usr/bin/env python3
"""
Module: accounting/accounting.py
"""

from typing import Any, List, Dict
from utils.error_handler import handle_error
from database.database import DBConnection, execute_sql, execute_sql_one


def validate_account(signal: Dict[str, Any], ctx: Any) -> bool:
    try:
        min_free_balance = ctx.config.get("min_free_balance_threshold", 0.001)
        with DBConnection(ctx.db_pool) as conn:
            row = execute_sql_one(
                conn,
                "SELECT balance, used_balance FROM account WHERE exchange = ?",
                [signal["exchange"]]
            )
            if row:
                free_balance = row["balance"] - row["used_balance"]
                return free_balance >= min_free_balance
            return False
    except Exception as e:
        handle_error(e, context="Accounting.validate_account", logger=ctx.logger)
        return False


def get_free_balance(exchange: str, ctx: Any) -> float:
    try:
        with DBConnection(ctx.db_pool) as conn:
            row = execute_sql_one(
                conn,
                "SELECT balance, used_balance FROM account WHERE exchange = ?",
                [exchange]
            )
            if row:
                return row["balance"] - row["used_balance"]
            return 0.0
    except Exception as e:
        handle_error(e, context="Accounting.get_free_balance", logger=ctx.logger)
        return 0.0


def record_new_trade(
    order: Dict[str, Any],
    signal: Dict[str, Any],
    ev: float,
    kelly_frac: float,
    position_size: float,
    ctx: Any,
    trade_source: str = "real"
) -> None:
    try:
        with DBConnection(ctx.db_pool) as conn:
            sql_ins = (
                "INSERT INTO trades (id, symbol, timeframe, trade_source, direction, "
                "entry_price, sl, tp, entry_time, exchange, position_size) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)"
            )
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
                position_size
            ]
            execute_sql(conn, sql_ins, params_ins)

            if trade_source == "real":
                upd_sql = (
                    "UPDATE account SET used_balance = used_balance + ? "
                    "WHERE exchange = ? AND used_balance + ? <= balance"
                )
                r = conn.cursor().execute(upd_sql, [position_size, signal["exchange"], position_size])
                conn.commit()
                if r.rowcount < 1:
                    raise ValueError("Insufficient free balance for this trade.")
    except Exception as e:
        handle_error(e, context="Accounting.record_new_trade", logger=ctx.logger)


def update_trade_result(trade_id: str, net_pnl: float, ctx: Any) -> None:
    try:
        with DBConnection(ctx.db_pool) as conn:
            sql_sel = "SELECT exchange, trade_source, position_size FROM trades WHERE id = ?"
            sel_row = execute_sql_one(conn, sql_sel, [trade_id])
            if sel_row:
                sql_upd = (
                    "UPDATE trades "
                    "SET close_time = datetime('now'), result = ?, close_reason = 'closed' "
                    "WHERE id = ?"
                )
                execute_sql(conn, sql_upd, [net_pnl, trade_id])
                if sel_row["trade_source"] == "real":
                    upd_sql = (
                        "UPDATE account SET used_balance = used_balance - ? "
                        "WHERE exchange = ? AND used_balance >= ?"
                    )
                    conn.cursor().execute(
                        upd_sql,
                        [sel_row["position_size"], sel_row["exchange"], sel_row["position_size"]]
                    )
                    conn.commit()
    except Exception as e:
        handle_error(e, context="Accounting.update_trade_result", logger=ctx.logger)


def update_trade_stop(trade_id: str, new_sl: float, ctx: Any) -> None:
    try:
        with DBConnection(ctx.db_pool) as conn:
            sql = "UPDATE trades SET sl = ? WHERE id = ?"
            execute_sql(conn, sql, [new_sl, trade_id])
    except Exception as e:
        handle_error(e, context="Accounting.update_trade_stop", logger=ctx.logger)


def fetch_open_trades(ctx: Any) -> List[Dict[str, Any]]:
    try:
        with DBConnection(ctx.db_pool) as conn:
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


def update_daily_performance(ctx: Any) -> None:
    try:
        with DBConnection(ctx.db_pool) as conn:
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
        handle_error(e, context="Accounting.update_daily_performance", logger=ctx.logger)

def log_performance_summary(ctx: Any) -> None:
    """Log daily performance summary with better formatting"""
    try:
        with DBConnection(ctx.db_pool) as conn:
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
