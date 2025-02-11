#!/usr/bin/env python3
"""
Module: startup/system_init.py
Production system initialization and verification
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json
import os
from decimal import Decimal
import psutil

from utils.error_handler import handle_error
from database.database import DBConnection, execute_sql

class SystemInitializer:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.startup_checks_passed = False
        self.initialization_errors: List[str] = []
        self.logger = ctx.logger or logging.getLogger(__name__)

    async def initialize_system(self) -> bool:
        """Complete system initialization sequence"""
        try:
            self.logger.info("Starting system initialization...")
            
            steps = [
                ("Configuration verification", self.verify_configuration),
                ("Component initialization", self.initialize_components),
                ("Database verification", self.verify_database),
                ("Exchange initialization", self.initialize_exchange),
                ("Trading system initialization", self.initialize_trading_system),
                ("Startup checks", self.run_startup_checks)
            ]
            
            for step_name, step_func in steps:
                self.logger.info(f"Starting {step_name}...")
                if not await step_func() if asyncio.iscoroutinefunction(step_func) else step_func():
                    self.logger.error(f"{step_name} failed")
                    return False
                self.logger.info(f"{step_name} completed successfully")

            self.startup_checks_passed = True
            self.logger.info("System initialization complete - ready for trading")
            return True
            
        except Exception as e:
            handle_error(e, "SystemInitializer.initialize_system", logger=self.logger)
            return False

    def verify_configuration(self) -> bool:
        """Verify all required configuration settings"""
        try:
            required_settings = {
                'timeframe': str,
                'market_list': list,
                'risk_factor': (float, Decimal),
                'max_daily_loss': (float, Decimal),
                'max_drawdown_pct': (float, Decimal),
                'emergency_stop_pct': (float, Decimal),
                'kelly_scaling': (float, Decimal),
                'max_position_size': (float, Decimal),
                'min_position_size': (float, Decimal),
                'exchange': str,
                'commission_rate': (float, Decimal),
                'database': dict
            }
            
            for setting, expected_type in required_settings.items():
                if setting not in self.ctx.config:
                    self.logger.error(f"Missing required setting: {setting}")
                    return False
                    
                value = self.ctx.config[setting]
                if not isinstance(value, expected_type if isinstance(expected_type, type) else expected_type[0]):
                    self.logger.error(f"Invalid type for {setting}: expected {expected_type}, got {type(value)}")
                    return False

            return True
            
        except Exception as e:
            handle_error(e, "SystemInitializer.verify_configuration", logger=self.logger)
            return False

    async def initialize_components(self) -> bool:
        """Initialize system components"""
        try:
            components = [
                ('health_monitor', self.ctx.health_monitor_class),
                ('position_manager', self.ctx.position_manager_class),
                ('order_manager', self.ctx.order_manager_class),
                ('risk_manager', self.ctx.risk_manager_class),
                ('circuit_breaker', self.ctx.circuit_breaker_class)
            ]
            
            for name, component_class in components:
                try:
                    setattr(self.ctx, name, component_class(self.ctx))
                    self.logger.info(f"Initialized {name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {name}: {str(e)}")
                    return False
                    
            return True
            
        except Exception as e:
            handle_error(e, "SystemInitializer.initialize_components", logger=self.logger)
            return False

    async def verify_database(self) -> bool:
        """Verify database schema and connectivity"""
        try:
            required_tables = [
                'account',
                'trades',
                'candles',
                'bot_performance',
                'balance_transactions'
            ]
            
            with self.ctx.db_pool.connection() as conn:
                existing_tables = execute_sql(
                    conn,
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                existing_tables = [t[0] for t in existing_tables]
                
                missing = set(required_tables) - set(existing_tables)
                if missing:
                    self.logger.error(f"Missing required tables: {', '.join(missing)}")
                    return False
                    
                for table in required_tables:
                    try:
                        execute_sql(conn, f"SELECT * FROM {table} LIMIT 1")
                    except Exception as e:
                        self.logger.error(f"Failed to query table {table}: {str(e)}")
                        return False
                        
            return True
            
        except Exception as e:
            handle_error(e, "SystemInitializer.verify_database", logger=self.logger)
            return False

    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection"""
        try:
            if self.ctx.config.get('paper_mode', True):
                self.logger.info("Running in paper trading mode")
                return True
                
            self.ctx.exchange = await self.ctx.exchange_class(self.ctx)
            if not self.ctx.exchange:
                self.logger.error("Failed to initialize exchange connection")
                return False
                
            # Verify API connectivity
            if not await self.ctx.exchange.check_connection():
                self.logger.error("Failed to verify exchange API connectivity")
                return False
                
            return True
            
        except Exception as e:
            handle_error(e, "SystemInitializer.initialize_exchange", logger=self.logger)
            return False

    async def initialize_trading_system(self) -> bool:
        """Initialize trading system components"""
        try:
            # Initialize market data system
            self.ctx.market_data = self.ctx.market_data_class(self.ctx)
            
            # Load initial market data
            if not await self.ctx.market_data.load_initial_data():
                self.logger.error("Failed to load initial market data")
                return False
                
            # Initialize ML models if configured
            if self.ctx.config.get('use_ml_signals', True):
                if not await self.ctx.ml_signal.initialize_models():
                    self.logger.error("Failed to initialize ML models")
                    return False
                    
            # Initialize GA system if configured
            if self.ctx.config.get('use_ga_signals', True):
                if not await self.ctx.ga_synergy.initialize_population():
                    self.logger.error("Failed to initialize GA population")
                    return False
                    
            return True
            
        except Exception as e:
            handle_error(e, "SystemInitializer.initialize_trading_system", logger=self.logger)
            return False

    async def run_startup_checks(self) -> bool:
        """Run comprehensive system startup checks"""
        try:
            checks = [
                self._check_disk_space(),
                self._check_memory(),
                await self._check_market_data(),
                await self._check_order_execution(),
                self._check_risk_limits()
            ]
            
            return all(checks)
            
        except Exception as e:
            handle_error(e, "SystemInitializer.run_startup_checks", logger=self.logger)
            return False

    def get_initialization_status(self) -> Dict[str, Any]:
        """Get system initialization status"""
        return {
            'initialization_complete': self.startup_checks_passed,
            'errors': self.initialization_errors,
            'components': {
                'database': hasattr(self.ctx, 'db'),
                'exchange': hasattr(self.ctx, 'exchange'),
                'market_data': hasattr(self.ctx, 'market_data'),
                'position_manager': hasattr(self.ctx, 'position_manager'),
                'risk_manager': hasattr(self.ctx, 'risk_manager'),
                'circuit_breaker': hasattr(self.ctx, 'circuit_breaker')
            }
        }