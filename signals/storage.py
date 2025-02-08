#!/usr/bin/env python3
"""
signals/storage.py - Database operations for trading rules
"""
import uuid
import json
import datetime
from typing import Dict, Any, Optional
from database.database import DBConnection, execute_sql, execute_sql_one
from utils.error_handler import handle_error

def store_rule(rule: Dict[str, Any], ctx: Any) -> None:
    """Store trading rule in database if it's better than current best"""
    if not ctx.config.get("store_rules", True):
        return
        
    try:
        current_fitness = rule.get("fitness", float("-inf"))
        if current_fitness is None:
            return
            
        with DBConnection(ctx.db_pool) as conn:
            row = execute_sql_one(
                conn,
                "SELECT MAX(fitness) as best_fitness FROM ga_rules",
                []
            )
            best_fitness = row["best_fitness"] if row and row["best_fitness"] is not None else float("-inf")
            
            if current_fitness > best_fitness:
                rule_id = str(uuid.uuid4())
                rule_json = json.dumps(rule)
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                
                sql = """
                    INSERT INTO ga_rules 
                    (id, chromosome_json, fitness, date_created)
                    VALUES (?, ?, ?, ?)
                """
                execute_sql(conn, sql, [rule_id, rule_json, current_fitness, timestamp])
                
                ctx.logger.info(f"Stored new best rule with fitness {current_fitness:.4f}")
                
    except Exception as e:
        handle_error(e, "storage.store_rule", logger=ctx.logger)

def load_best_rule(ctx: Any) -> Optional[Dict[str, Any]]:
    """Load best performing rule from database"""
    try:
        with DBConnection(ctx.db_pool) as conn:
            row = execute_sql_one(
                conn,
                """
                SELECT chromosome_json
                FROM ga_rules
                ORDER BY fitness DESC
                LIMIT 1
                """,
                []
            )
            
            if row and row["chromosome_json"]:
                return json.loads(row["chromosome_json"])
                
    except Exception as e:
        handle_error(e, "storage.load_best_rule", logger=ctx.logger)
        
    return None