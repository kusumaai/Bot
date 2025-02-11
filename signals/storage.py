#!/usr/bin/env python3
"""
signals/storage.py - Database operations for trading rules
"""
import uuid
import json
import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
import logging

from database.database import DBConnection, execute_sql, execute_sql_one
from utils.error_handler import handle_error
from signals.trading_types import TradingRule
from database.queries import DatabaseQueries
from utils.error_handler import DatabaseError

def store_rule(rule: Dict[str, Any], ctx: Any) -> bool:
    """Store trading rule in database if it's better than current best"""
    if not ctx.config.get("store_rules", True):
        return False
        
    try:
        current_fitness = Decimal(str(rule.get("fitness", float("-inf"))))
        if current_fitness is None or current_fitness <= 0:
            return False
            
        with DBConnection(ctx.db_pool) as conn:
            # Get current best fitness
            row = execute_sql_one(
                conn,
                """
                SELECT MAX(fitness) as best_fitness 
                FROM ga_rules 
                WHERE date_created >= datetime('now', '-7 days')
                """,
                []
            )
            best_fitness = (
                Decimal(str(row["best_fitness"])) 
                if row and row["best_fitness"] is not None 
                else Decimal("-inf")
            )
            
            # Store if better than current best
            if current_fitness > best_fitness:
                rule_id = str(uuid.uuid4())
                rule_json = json.dumps(rule)
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                
                sql = """
                    INSERT INTO ga_rules 
                    (id, chromosome_json, fitness, date_created)
                    VALUES (?, ?, ?, ?)
                """
                execute_sql(conn, sql, [rule_id, rule_json, float(current_fitness), timestamp])
                
                # Cleanup old rules
                cleanup_sql = """
                    DELETE FROM ga_rules 
                    WHERE date_created < datetime('now', '-30 days')
                    OR fitness < ?
                """
                execute_sql(conn, cleanup_sql, [float(best_fitness * Decimal("0.5"))])
                
                ctx.logger.info(f"Stored new best rule with fitness {float(current_fitness):.4f}")
                return True
                
            return False
            
    except Exception as e:
        handle_error(e, "storage.store_rule", logger=ctx.logger)
        return False

def load_best_rule(ctx: Any) -> Optional[Dict[str, Any]]:
    """Load best performing rule from database with validation"""
    try:
        with DBConnection(ctx.db_pool) as conn:
            row = execute_sql_one(
                conn,
                """
                SELECT chromosome_json, fitness, date_created
                FROM ga_rules
                WHERE date_created >= datetime('now', '-7 days')
                ORDER BY fitness DESC
                LIMIT 1
                """,
                []
            )
            
            if row and row["chromosome_json"]:
                rule = json.loads(row["chromosome_json"])
                ctx.logger.info(
                    f"Loaded rule with fitness {row['fitness']:.4f} "
                    f"from {row['date_created']}"
                )
                return rule
                
            ctx.logger.warning("No valid rules found in database")
            return None
            
    except Exception as e:
        handle_error(e, "storage.load_best_rule", logger=ctx.logger)
        return None

def get_rule_stats(ctx: Any) -> Dict[str, Any]:
    """Get statistics about stored rules"""
    try:
        with DBConnection(ctx.db_pool) as conn:
            stats = execute_sql_one(
                conn,
                """
                SELECT 
                    COUNT(*) as total_rules,
                    AVG(fitness) as avg_fitness,
                    MAX(fitness) as max_fitness,
                    MIN(fitness) as min_fitness,
                    strftime('%Y-%m-%d %H:%M:%S', MAX(date_created)) as latest_rule
                FROM ga_rules
                WHERE date_created >= datetime('now', '-7 days')
                """,
                []
            )
            return dict(stats) if stats else {}
            
    except Exception as e:
        handle_error(e, "storage.get_rule_stats", logger=ctx.logger)
        return {}

class SignalStorage:
    def __init__(self, db_queries: DatabaseQueries, logger: logging.Logger):
        self.db = db_queries
        self.logger = logger
        
    async def store_signal(
        self,
        symbol: str,
        signal_type: str,
        direction: str,
        strength: float,
        metadata: Optional[Dict[str, Any]] = None,
        expiry: Optional[datetime] = None
    ) -> int:
        try:
            query = """
                INSERT INTO signals (
                    symbol, signal_type, direction, strength,
                    metadata, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = [
                symbol,
                signal_type,
                direction,
                strength,
                json.dumps(metadata) if metadata else None,
                datetime.utcnow().timestamp(),
                expiry.timestamp() if expiry else None
            ]
            
            result = await self.db.execute(query, params)
            return result.lastrowid
            
        except Exception as e:
            raise DatabaseError(f"Failed to store signal: {str(e)}")
    
    async def get_active_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            conditions = ["expires_at IS NULL OR expires_at > ?"]
            params = [datetime.utcnow().timestamp()]
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if signal_type:
                conditions.append("signal_type = ?")
                params.append(signal_type)
            
            query = f"""
                SELECT * FROM signals
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
            """
            
            return await self.db.execute(query, params, fetch=True)
            
        except Exception as e:
            raise DatabaseError(f"Failed to fetch signals: {str(e)}")
    
    async def update_signal_status(
        self,
        signal_id: int,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        try:
            query = """
                UPDATE signals
                SET status = ?,
                    metadata = json_patch(metadata, ?),
                    updated_at = ?
                WHERE id = ?
            """
            
            params = [
                status,
                json.dumps(metadata or {}),
                datetime.utcnow().timestamp(),
                signal_id
            ]
            
            await self.db.execute(query, params)
            
        except Exception as e:
            raise DatabaseError(f"Failed to update signal: {str(e)}")