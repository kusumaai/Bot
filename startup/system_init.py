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
from dataclasses import dataclass

from utils.error_handler import handle_error, handle_error_async, init_error_handler
from database.database import DBConnection, execute_sql
from utils.health_monitor import HealthMonitor, ComponentHealth
from trading.portfolio_manager import PortfolioManager
from execution.order_manager import OrderManager
from execution.market_data import MarketData
from execution.exchange_interface import ExchangeInterface
from trading.circuit_breaker import CircuitBreaker
from trading.ratchet import RatchetManager
from config.risk_config import RiskConfig

@dataclass
class HealthStatus:
    timestamp: float
    is_healthy: bool
    warnings: List[str]
    errors: List[str]
    components: Dict[str, 'ComponentHealth']

class SystemInitializer:
    def __init__(self, ctx: TradingContext):
        self.ctx = ctx
        self.startup_checks_passed = False
        self.initialization_errors: List[str] = []
        self.logger = ctx.logger
        
        # Initialize error handler with database path
        self.risk_config = RiskConfig.from_config(ctx.config.get("risk", {}))
        self.db_connection = DBConnection(self.ctx.config.get("database", {}).get("path", "data/trading.db"))
        init_error_handler(self.ctx.config.get("database", {}).get("path", "data/trading.db"))

        self.health_monitor_class = HealthMonitor
        self.portfolio_manager_class = PortfolioManager
        self.order_manager_class = OrderManager
        self.market_data_class = MarketData
        self.circuit_breaker_class = CircuitBreaker
        self.ratchet_manager_class = RatchetManager
        # Add other component classes as needed

    async def initialize_system(self) -> bool:
        try:
            # Initialize all components in the recommended order
            self.health_monitor = self.health_monitor_class(self.ctx)
            self.portfolio_manager = self.portfolio_manager_class(self.risk_config, logger=self.logger)
            self.order_manager = self.order_manager_class(self.ctx)
            self.market_data = self.market_data_class(self.ctx)
            await self.market_data.initialize()
            self.circuit_breaker = self.circuit_breaker_class(self.ctx.db_queries, self.logger)
            await self.circuit_breaker.initialize()
            self.ratchet_manager = RatchetManager(self.ctx)
            await self.ratchet_manager.initialize()
            
            self.health_monitor.start_monitoring()
            self.startup_checks_passed = True
            return True
        except Exception as e:
            await handle_error_async(e, "SystemInitializer.initialize_system", self.logger)
            self.initialization_errors.append(str(e))
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
            components: List[Tuple[str, Any]] = [
                ('health_monitor', self.health_monitor_class),
                ('position_manager', self.portfolio_manager_class),
                ('order_manager', self.order_manager_class),
                ('market_data', self.market_data_class),
                ('circuit_breaker', self.circuit_breaker_class),
                ('ratchet_manager', self.ratchet_manager_class),
                # Add other components here
            ]
            
            for name, component_class in components:
                try:
                    component_instance = component_class(self.ctx)
                    setattr(self.ctx, name, component_instance)
                    
                    # If the component has an async initialize method, await it
                    if hasattr(component_instance, 'initialize') and asyncio.iscoroutinefunction(component_instance.initialize):
                        await component_instance.initialize()
                    
                    self.logger.info(f"Initialized {name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {name}: {str(e)}")
                    return False
                    
            return True
            
        except Exception as e:
            handle_error(e, "SystemInitializer.initialize_components", logger=self.logger)
            return False

    async def verify_database(self) -> bool:
        """Verify database tables with SQL injection protection"""
        try:
            required_tables = ['trades', 'positions', 'performance']
            placeholders = ', '.join('?' * len(required_tables))
            query = f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ({placeholders})"
            
            async with self.db_connection.get_connection() as conn:
                async with conn.execute(query, required_tables) as cursor:
                    existing = await cursor.fetchall()
                    missing = set(required_tables) - {row[0] for row in existing}
                    
                    if missing:
                        self.logger.error(f"Missing required tables: {missing}")
                        return False
                    return True
                        
        except Exception as e:
            await handle_error_async(e, "SystemInitializer.verify_database", self.logger)
            return False

    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection"""
        try:
            if self.ctx.config.get('paper_mode', True):
                self.logger.info("Running in paper trading mode")
                return True
                
            self.ctx.exchange_interface = self.ctx.exchange_interface  # Ensure exchange_interface is initialized
            if not self.ctx.exchange_interface.exchange:
                self.logger.error("Failed to initialize exchange connection")
                return False
                
            # Verify API connectivity
            if not await self.ctx.exchange_interface.exchange.ping():
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
                'circuit_breaker': hasattr(self.ctx, 'circuit_breaker'),
                'ratchet_manager': hasattr(self.ctx, 'ratchet_manager')
            }
        }