#!/usr/bin/env python3
"""
Module: startup/system_init.py
Production system initialization and verification
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import os

class SystemInitializer:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.startup_checks_passed = False
        self.initialization_errors: List[str] = []
        
    async def initialize_system(self) -> bool:
        """Complete system initialization sequence"""
        try:
            self.ctx.logger.info("Starting system initialization...")
            
            # Verify configuration
            if not self.verify_configuration():
                return False
                
            # Initialize components
            components_initialized = await self.initialize_components()
            if not components_initialized:
                return False
                
            # Verify database
            if not await self.verify_database():
                return False
                
            # Initialize exchange connection
            if not await self.initialize_exchange():
                return False
                
            # Load models and verify signals
            if not await self.initialize_trading_system():
                return False
                
            # Run startup checks
            if not await self.run_startup_checks():
                return False
                
            self.startup_checks_passed = True
            self.ctx.logger.info("System initialization complete - ready for trading")
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"System initialization failed: {str(e)}")
            return False

    def verify_configuration(self) -> bool:
        """Verify all required configuration settings"""
        try:
            required_settings = [
                # Core settings
                'timeframe',
                'market_list',
                'risk_factor',
                
                # Risk limits
                'max_daily_loss',
                'max_drawdown_pct',
                'emergency_stop_pct',
                
                # Position sizing
                'kelly_scaling',
                'max_position_size',
                'min_position_size',
                
                # Exchange settings
                'exchange',
                'commission_rate',
                
                # Database settings
                'database.path'
            ]
            
            missing = []
            for setting in required_settings:
                if '.' in setting:
                    parent, child = setting.split('.')
                    if parent not in self.ctx.config or \
                       child not in self.ctx.config[parent]:
                        missing.append(setting)
                elif setting not in self.ctx.config:
                    missing.append(setting)
                    
            if missing:
                self.ctx.logger.error(
                    f"Missing required configuration settings: {', '.join(missing)}"
                )
                return False
                
            # Validate numeric settings
            numeric_settings = [
                ('risk_factor', 0, 1),
                ('max_daily_loss', 0, None),
                ('max_drawdown_pct', 0, 100),
                ('kelly_scaling', 0, 1),
                ('max_position_size', 0, 1),
                ('commission_rate', 0, 0.01)
            ]
            
            for setting, min_val, max_val in numeric_settings:
                value = float(self.ctx.config[setting])
                if min_val is not None and value < min_val:
                    self.ctx.logger.error(
                        f"Setting {setting} value {value} below minimum {min_val}"
                    )
                    return False
                if max_val is not None and value > max_val:
                    self.ctx.logger.error(
                        f"Setting {setting} value {value} above maximum {max_val}"
                    )
                    return False
                    
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Configuration verification failed: {str(e)}")
            return False
            
    async def initialize_components(self) -> bool:
        """Initialize system components"""
        try:
            # Set up error handler first
            self.ctx.error_handler = self.ctx.error_handler_class(self.ctx)
            
            # Initialize database connection
            self.ctx.db = self.ctx.database_class(
                self.ctx.config['database']['path']
            )
            
            # Initialize core components
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
                    self.ctx.logger.info(f"Initialized {name}")
                except Exception as e:
                    self.ctx.logger.error(f"Failed to initialize {name}: {str(e)}")
                    return False
                    
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Component initialization failed: {str(e)}")
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
                # Check tables exist
                existing_tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                existing_tables = [t[0] for t in existing_tables]
                
                missing = set(required_tables) - set(existing_tables)
                if missing:
                    self.ctx.logger.error(
                        f"Missing required database tables: {', '.join(missing)}"
                    )
                    return False
                    
                # Verify table schemas
                for table in required_tables:
                    try:
                        conn.execute(f"SELECT * FROM {table} LIMIT 1")
                    except Exception as e:
                        self.ctx.logger.error(
                            f"Failed to query table {table}: {str(e)}"
                        )
                        return False
                        
                return True
                
        except Exception as e:
            self.ctx.logger.error(f"Database verification failed: {str(e)}")
            return False

    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection"""
        try:
            if self.ctx.config.get('paper_mode', True):
                self.ctx.logger.info("Running in paper trading mode")
                return True
                
            # Initialize exchange interface
            self.ctx.exchange = await self.ctx.exchange_class(self.ctx)
            if not self.ctx.exchange:
                self.ctx.logger.error("Failed to initialize exchange connection")
                return False
                
            # Verify API connectivity
            markets = await self.ctx.exchange.fetch_markets()
            if not markets:
                self.ctx.logger.error("Failed to fetch exchange markets")
                return False
                
            # Verify trading enabled for configured pairs
            for symbol in self.ctx.config['market_list']:
                if symbol not in markets:
                    self.ctx.logger.error(f"Market {symbol} not available on exchange")
                    return False
                    
                market = markets[symbol]
                if not market['active']:
                    self.ctx.logger.error(f"Market {symbol} is not active")
                    return False
                    
                # Store market limits
                self.ctx.config['min_trade_sizes'][symbol] = {
                    'min_notional': market.get('limits', {}).get('cost', {}).get('min', 0),
                    'min_qty': market.get('limits', {}).get('amount', {}).get('min', 0)
                }
                
            # Verify account balances
            balances = await self.ctx.exchange.fetch_balance()
            if not balances:
                self.ctx.logger.error("Failed to fetch account balances")
                return False
                
            # Update database account balances
            await self.update_account_balances(balances)
            
            self.ctx.logger.info("Exchange initialization complete")
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Exchange initialization failed: {str(e)}")
            return False

    async def initialize_trading_system(self) -> bool:
        """Initialize trading system components"""
        try:
            # Load ML models if configured
            if self.ctx.config.get('use_ml_signals', True):
                rf_model, xgb_model, trained_columns = self.ctx.ml_signal.load_models(self.ctx)
                
                if not rf_model and not xgb_model:
                    self.ctx.logger.error("Failed to load ML models")
                    return False
                    
                self.ctx.rf_model = rf_model
                self.ctx.xgb_model = xgb_model
                self.ctx.trained_columns = trained_columns
                
                self.ctx.logger.info("ML models loaded successfully")
                
            # Initialize GA population if configured
            if self.ctx.config.get('use_ga_signals', True):
                population = self.ctx.ga_synergy.initialize_population(self.ctx)
                if not population:
                    self.ctx.logger.error("Failed to initialize GA population")
                    return False
                    
                self.ctx.population = population
                self.ctx.logger.info("GA population initialized successfully")
                
            # Initialize market data system
            self.ctx.market_data = self.ctx.market_data_class(self.ctx)
            
            # Load initial market data
            market_data, dataframes = await self.ctx.market_data.load_market_data(
                self.ctx.config['market_list']
            )
            
            if not market_data or not dataframes:
                self.ctx.logger.error("Failed to load initial market data")
                return False
                
            self.ctx.logger.info("Trading system initialization complete")
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Trading system initialization failed: {str(e)}")
            return False

    async def update_account_balances(self, balances: Dict[str, Any]) -> None:
        """Update database account balances"""
        try:
            with self.ctx.db_pool.connection() as conn:
                for currency, balance in balances.items():
                    if balance['free'] > 0 or balance['used'] > 0:
                        conn.execute("""
                            INSERT OR REPLACE INTO account (
                                exchange, balance, used_balance, last_reconciled
                            ) VALUES (?, ?, ?, datetime('now'))
                        """, [
                            self.ctx.config['exchange'],
                            float(balance['free'] + balance['used']),
                            float(balance['used'])
                        ])
                conn.commit()
        except Exception as e:
            self.ctx.logger.error(f"Failed to update account balances: {str(e)}")
            raise

    async def run_startup_checks(self) -> bool:
        """Run comprehensive system startup checks"""
        try:
            checks = [
                self._check_disk_space(),
                self._check_memory(),
                await self._check_market_data(),
                await self._check_order_execution(),
                self._check_risk_limits(),
                await self._check_balance_reconciliation()
            ]
            
            return all(checks)
            
        except Exception as e:
            self.ctx.logger.error(f"Startup checks failed: {str(e)}")
            return False

    def _check_disk_space(self) -> bool:
        """Verify sufficient disk space"""
        try:
            import psutil
            
            disk = psutil.disk_usage(os.path.dirname(self.ctx.config['database']['path']))
            min_space_gb = self.ctx.config.get('min_disk_space_gb', 5)
            
            free_gb = disk.free / (1024 * 1024 * 1024)
            if free_gb < min_space_gb:
                self.ctx.logger.error(
                    f"Insufficient disk space: {free_gb:.1f}GB free, "
                    f"minimum {min_space_gb}GB required"
                )
                return False
                
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Disk space check failed: {str(e)}")
            return False

    def _check_memory(self) -> bool:
        """Verify sufficient memory"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            min_memory_pct = 100 - self.ctx.config.get('max_memory_usage_pct', 90)
            
            if memory.percent > min_memory_pct:
                self.ctx.logger.error(
                    f"Insufficient memory: {memory.percent}% used, "
                    f"maximum {min_memory_pct}% allowed"
                )
                return False
                
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Memory check failed: {str(e)}")
            return False

    async def _check_market_data(self) -> bool:
        """Verify market data freshness"""
        try:
            for symbol in self.ctx.config['market_list']:
                with self.ctx.db_pool.connection() as conn:
                    latest = conn.execute("""
                        SELECT MAX(timestamp) as ts
                        FROM candles
                        WHERE symbol = ?
                    """, [symbol]).fetchone()
                    
                    if not latest or not latest[0]:
                        self.ctx.logger.error(f"No market data found for {symbol}")
                        return False
                        
                    # Check data freshness
                    data_age = datetime.now().timestamp() - latest[0]/1000
                    max_age = self.ctx.config.get('max_data_age_hours', 24) * 3600
                    
                    if data_age > max_age:
                        self.ctx.logger.error(
                            f"Market data for {symbol} is too old: "
                            f"{data_age/3600:.1f} hours"
                        )
                        return False
                        
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Market data check failed: {str(e)}")
            return False

    async def _check_order_execution(self) -> bool:
        """Verify order execution"""
        if self.ctx.config.get('paper_mode', True):
            return True
            
        try:
            # Place and cancel test order
            symbol = self.ctx.config['market_list'][0]
            min_qty = self.ctx.config['min_trade_sizes'][symbol]['min_qty']
            
            order = await self.ctx.exchange.create_limit_order(
                symbol=symbol,
                side='buy',
                amount=min_qty,
                price=0.01  # Far from market price
            )
            
            if not order:
                self.ctx.logger.error("Failed to place test order")
                return False
                
            # Cancel order
            cancelled = await self.ctx.exchange.cancel_order(order['id'], symbol)
            if not cancelled:
                self.ctx.logger.error("Failed to cancel test order")
                return False
                
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Order execution check failed: {str(e)}")
            return False

    def _check_risk_limits(self) -> bool:
        """Verify risk limits configuration"""
        try:
            required_limits = [
                ('max_position_size', 0.5),
                ('max_daily_loss', 1000),
                ('max_drawdown_pct', 20),
                ('emergency_stop_pct', 5)
            ]
            
            for limit, max_value in required_limits:
                value = float(self.ctx.config.get(limit, 0))
                if value <= 0 or value > max_value:
                    self.ctx.logger.error(
                        f"Invalid {limit} configuration: {value} "
                        f"(max allowed: {max_value})"
                    )
                    return False
                    
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Risk limits check failed: {str(e)}")
            return False

    async def _check_balance_reconciliation(self) -> bool:
        """Verify balance reconciliation"""
        try:
            if self.ctx.config.get('paper_mode', True):
                return True
                
            # Get exchange balances
            exchange_balances = await self.ctx.exchange.fetch_balance()
            if not exchange_balances:
                return False
                
            # Compare with database
            with self.ctx.db_pool.connection() as conn:
                db_balances = conn.execute("""
                    SELECT exchange, balance, used_balance
                    FROM account
                    WHERE exchange = ?
                """, [self.ctx.config['exchange']]).fetchall()
                
                for row in db_balances:
                    currency = row[0]
                    if currency in exchange_balances:
                        ex_total = exchange_balances[currency]['total']
                        db_total = row[1]
                        
                        if abs(ex_total - db_total) > 0.0001:
                            self.ctx.logger.error(
                                f"Balance mismatch for {currency}: "
                                f"Exchange={ex_total}, DB={db_total}"
                            )
                            return False
                            
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Balance reconciliation check failed: {str(e)}")
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
            },
            'model_status': {
                'rf_model': hasattr(self.ctx, 'rf_model'),
                'xgb_model': hasattr(self.ctx, 'xgb_model'),
                'ga_population': hasattr(self.ctx, 'population')
            }
        }