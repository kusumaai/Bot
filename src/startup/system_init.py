#!/usr/bin/env python3
# src/startup/system_init.py   
"""
Module: startup/system_init.py
System initialization and component setup
"""
#import required modules
from typing import List, Dict, Any
import logging
from pathlib import Path
from utils.error_handler import init_error_handler
from database.database import DatabaseConnection
from config.risk_config import RiskConfig
from utils.logger import setup_logging
#system initializer class that initializes the system by running the system initialization and startup checks using the run_system_init and run_startup_checks methods  
class SystemInitializer:
    def __init__(self, ctx: Any):
        """Initialize the SystemInitializer class."""
        self.ctx = ctx
        self.logger = setup_logging(name="SystemInit", level="INFO")
        self.startup_checks_passed = False
        self.initialization_errors: List[str] = []
        #initialize database and error handler
        try:
            # Initialize database and error handler first
            db_path = self.ctx.config.get("database", {}).get("path", "data/trading.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.db_connection = DatabaseConnection(db_path)
            init_error_handler(db_path)
            
            # Load risk configuration
            self.risk_config = RiskConfig.from_config(ctx.config.get("risk", {}))
            
            # Component initialization will be handled by TradingContext
            self.startup_checks_passed = True
            
        except Exception as e:
            self.initialization_errors.append(f"System initialization failed: {str(e)}")
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise
    #run system initialization checks
    async def run_startup_checks(self) -> bool:
        """Run system startup checks"""
        try:
            if not self.startup_checks_passed:
                self.logger.error("System initialization failed, cannot run startup checks")
                return False
                
            # Check database connection
            if not await self.db_connection.test_connection():
                self.initialization_errors.append("Database connection failed")
                return False
                
            # Check disk space
            if not self._check_disk_space():
                self.initialization_errors.append("Insufficient disk space")
                return False
                
            return True
            
        except Exception as e:
            self.initialization_errors.append(f"Startup checks failed: {str(e)}")
            self.logger.error(f"Failed to run startup checks: {str(e)}")
            return False
    #check  if sufficient disk space is available
    def _check_disk_space(self) -> bool:
        """Check if sufficient disk space is available"""
        try:
            min_space_mb = 1000  # 1GB minimum
            db_path = Path(self.ctx.config.get("database", {}).get("path", "data/trading.db"))
            free_space = db_path.parent.stat().st_free / (1024 * 1024)  # Convert to MB
            
            return free_space >= min_space_mb
            
        except Exception as e:
            self.logger.error(f"Failed to check disk space: {str(e)}")
            return False
        
    #initialize the system by running the system initialization and startup checks using the run_system_init and run_startup_checks methods
    async def initialize(self):
        """Initialize the system"""
        try:
            # Run system initialization
            if not await self.run_system_init():
                self.logger.error("System initialization failed")
                return False

            # Run startup checks
            if not await self.run_startup_checks():
                self.logger.error("Startup checks failed")
                return False

            self.logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            self.initialization_errors.append(f"System initialization failed: {str(e)}")
            return False
