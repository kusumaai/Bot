#!/usr/bin/env python3
"""
Database migration script
"""
import sqlite3
import logging
import os
import shutil
from pathlib import Path
from contextlib import closing

def check_db_integrity(db_path: str) -> bool:
    """Check database integrity"""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            return result == "ok"
    except Exception as e:
        logging.error(f"Integrity check failed: {e}")
        return False

def apply_migration(db_path: str, migration_path: str) -> bool:
    """Apply database migration safely"""
    logger = logging.getLogger(__name__)
    conn = None
    
    try:
        # Ensure the database file exists
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            return False
            
        # Check integrity first
        if not check_db_integrity(db_path):
            logger.error("Database integrity check failed")
            return False
            
        # Read migration SQL
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        # Create backup
        backup_path = f"{db_path}.backup"
        logger.info(f"Creating backup at {backup_path}")
        
        # Ensure any existing connections are closed before file operations
        if conn:
            conn.close()
            conn = None
            
        shutil.copy2(db_path, backup_path)
        
        # Apply migration
        logger.info("Applying migration...")
        conn = sqlite3.connect(db_path)
        conn.executescript(migration_sql)
        
        # Verify migration
        cursor = conn.cursor()
        cursor.execute("""
            SELECT CASE 
                WHEN EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='candles_backup')
                AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='account_balance')
                THEN 'Success' 
                ELSE 'Failed' 
            END as migration_status
        """)
        status = cursor.fetchone()[0]
        
        if status == 'Success':
            logger.info("Migration completed successfully")
            if conn:
                conn.close()
                conn = None
            return True
        else:
            logger.error("Migration verification failed")
            if conn:
                conn.close()
                conn = None
            # Restore backup
            shutil.copy2(backup_path, db_path)
            return False
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if conn:
            conn.close()
            conn = None
        if os.path.exists(backup_path):
            logger.info("Restoring from backup...")
            try:
                shutil.copy2(backup_path, db_path)
            except PermissionError:
                logger.error("Could not restore backup - file is locked")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Get paths
    current_dir = Path(__file__).parent
    db_path = current_dir.parent / "data" / "candles.db"
    migration_path = current_dir / "migrations" / "001_expand_schema.sql"
    
    # First check if database is corrupted
    if not check_db_integrity(str(db_path)):
        print("Database is corrupted. Please restore from a backup before migrating.")
        exit(1)
    
    # Apply migration
    success = apply_migration(str(db_path), str(migration_path))
    exit(0 if success else 1)
