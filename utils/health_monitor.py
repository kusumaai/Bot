#!/usr/bin/env python3
"""
Module: utils/health_monitor.py
Production health monitoring system
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass, field
import logging
from decimal import Decimal
from utils.error_handler import handle_error, handle_error_async
from utils.numeric import NumericHandler
from collections import deque
import aiohttp

@dataclass
class ComponentHealth:
    healthy: bool
    last_check: float
    latency: float
    error_count: int
    last_error: Optional[str]
    degraded: bool = False
    last_degradation: Optional[float] = None

@dataclass
class HealthStatus:
    timestamp: float
    is_healthy: bool
    warnings: List[str]
    errors: List[str]
    components: Dict[str, ComponentHealth]
    system_metrics: Dict[str, Any]

class HealthMonitor:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.nh = NumericHandler()
        self._component_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._metric_lock = asyncio.Lock()
        
        # Use deque for fixed-size history
        self.latency_history: Dict[str, deque] = {}
        self.max_history = 1000
        
        self.status = {
            'system': True,
            'database': True,
            'exchange': True,
            'market_data': True
        }
        self.last_check = {}
        self.error_counts: Dict[str, int] = {}
        self.check_interval = 60  # seconds
        
        # Initialize status
        self.status = HealthStatus(
            timestamp=time.time(),
            is_healthy=True,
            warnings=[],
            errors=[],
            components={},
            system_metrics={}
        )
        
        # Component tracking
        self.components: Dict[str, ComponentHealth] = {
            'database': ComponentHealth(True, time.time(), 0, 0, None),
            'exchange': ComponentHealth(True, time.time(), 0, 0, None),
            'order_manager': ComponentHealth(True, time.time(), 0, 0, None),
            'position_manager': ComponentHealth(True, time.time(), 0, 0, None),
            'risk_manager': ComponentHealth(True, time.time(), 0, 0, None),
            'system': ComponentHealth(True, time.time(), 0, 0, None)
        }
        
        # Performance tracking
        self.error_window = timedelta(minutes=5)
        self.degradation_threshold = Decimal("2.0")  # 2x baseline is considered degraded
        
        # System thresholds
        self.thresholds = {
            'memory': Decimal(str(ctx.config.get("memory_threshold_pct", 90))),
            'disk': Decimal(str(ctx.config.get("disk_threshold_pct", 90))),
            'cpu': Decimal(str(ctx.config.get("cpu_threshold_pct", 80))),
            'max_errors': ctx.config.get("max_component_errors", 5),
            'max_latency': Decimal(str(ctx.config.get("max_latency_sec", 5)))
        }
        
        # Start monitoring
        asyncio.create_task(self.start_monitoring())

    async def start_monitoring(self):
        """Start the health monitoring loop with proper error handling"""
        while True:
            try:
                async with self._component_lock:
                    await self.check_system_health()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                await handle_error_async(e, "HealthMonitor.start_monitoring", self.logger)
                await asyncio.sleep(5)  # Back off on error

    async def check_system_health(self) -> Dict[str, bool]:
        """Comprehensive system health check"""
        async with self._component_lock:
            try:
                # System resources check
                system_ok = await self._check_system_resources()
                
                # Database connection check
                db_ok = await self._check_database()
                
                # Exchange connection check
                exchange_ok = await self._check_exchange()
                
                # Market data freshness check
                market_data_ok = await self._check_market_data()
                
                self.status.update({
                    'system': system_ok,
                    'database': db_ok,
                    'exchange': exchange_ok,
                    'market_data': market_data_ok
                })
                
                self.last_check = datetime.utcnow()
                return self.status
                
            except Exception as e:
                self.ctx.logger.error(f"Health check failed: {e}")
                return {k: False for k in self.status}
                
    async def _check_system_resources(self) -> bool:
        """Check system CPU, memory, and disk"""
        try:
            # CPU usage check
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 80:  # 80% threshold
                self.ctx.logger.warning(f"High CPU usage: {cpu_percent}%")
                return False
                
            # Memory usage check
            memory = psutil.virtual_memory()
            if memory.percent > 85:  # 85% threshold
                self.ctx.logger.warning(f"High memory usage: {memory.percent}%")
                return False
                
            # Disk space check
            disk = psutil.disk_usage('/')
            if disk.percent > 90:  # 90% threshold
                self.ctx.logger.warning(f"Low disk space: {disk.percent}%")
                return False
                
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Resource check failed: {e}")
            return False
            
    async def _check_database(self) -> bool:
        """Verify database connection and performance"""
        try:
            start_time = time.time()
            await self.ctx.db.ping()
            response_time = time.time() - start_time
            
            if response_time > 1.0:  # 1 second threshold
                self.ctx.logger.warning(f"Slow database response: {response_time}s")
                return False
                
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Database check failed: {e}")
            return False
            
    async def _check_exchange(self) -> bool:
        """Verify exchange connectivity"""
        try:
            start_time = time.time()
            await self.ctx.exchange_interface.fetch_market_info("BTC/USDT")
            response_time = time.time() - start_time
            
            if response_time > 2.0:  # 2 second threshold
                self.ctx.logger.warning(f"Slow exchange response: {response_time}s")
                return False
                
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Exchange check failed: {e}")
            return False
            
    async def _check_market_data(self) -> bool:
        """Verify market data freshness"""
        try:
            for symbol in self.ctx.config['market_list']:
                last_update = self.ctx.market_data.last_update.get(symbol)
                if not last_update or time.time() - last_update > 300:  # 5 minutes
                    self.ctx.logger.warning(f"Stale market data for {symbol}")
                    return False
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Market data check failed: {e}")
            return False

    async def check_database(self) -> Tuple[bool, float, Optional[str]]:
        """Check database connectivity"""
        start = time.time()
        try:
            async with self.ctx.db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True, time.time() - start, None
        except Exception as e:
            handle_error(e, "HealthMonitor.check_database", logger=self.logger)
            return False, time.time() - start, str(e)

    async def check_exchange(self) -> Tuple[bool, float, Optional[str]]:
        """Check exchange connectivity"""
        start = time.time()
        try:
            await self.ctx.exchange.ping()
            return True, time.time() - start, None
        except Exception as e:
            handle_error(e, "HealthMonitor.check_exchange", logger=self.logger)
            return False, time.time() - start, str(e)

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory_used_pct': Decimal(str(memory.percent)),
                'memory_healthy': memory.percent < float(self.thresholds['memory']),
                'disk_used_pct': Decimal(str(disk.percent)),
                'disk_healthy': disk.percent < float(self.thresholds['disk']),
                'cpu_used_pct': Decimal(str(cpu_percent)),
                'cpu_healthy': cpu_percent < float(self.thresholds['cpu'])
            }
        except Exception as e:
            handle_error(e, "HealthMonitor.check_system_resources", logger=self.logger)
            return {
                'memory_used_pct': Decimal("0"),
                'memory_healthy': False,
                'disk_used_pct': Decimal("0"),
                'disk_healthy': False,
                'cpu_used_pct': Decimal("0"),
                'cpu_healthy': False
            }

    def update_component(
        self,
        component: str,
        healthy: bool,
        latency: float,
        error: Optional[str] = None
    ) -> None:
        """Update component health status"""
        try:
            now = time.time()
            
            if component not in self.components:
                self.components[component] = ComponentHealth(
                    healthy=healthy,
                    last_check=now,
                    latency=latency,
                    error_count=0,
                    last_error=None
                )
                
            comp = self.components[component]
            comp.healthy = healthy
            comp.last_check = now
            comp.latency = latency
            
            if error:
                comp.error_count += 1
                comp.last_error = error
                
            # Track latency history
            if component not in self.latency_history:
                self.latency_history[component] = deque(maxlen=self.max_history)
            self.latency_history[component].append(latency)
            
            # Check for degradation
            if len(self.latency_history[component]) >= 100:
                baseline = np.median(self.latency_history[component])
                comp.degraded = latency > float(self.degradation_threshold) * baseline
                if comp.degraded:
                    comp.last_degradation = now
                    self.logger.warning(
                        f"Performance degradation detected for {component}: "
                        f"Current={latency:.3f}s, Baseline={baseline:.3f}s"
                    )
                    
        except Exception as e:
            handle_error(e, "HealthMonitor.update_component", logger=self.logger)

    def should_emergency_shutdown(self) -> bool:
        """Determine if emergency shutdown needed"""
        try:
            # Critical components must be healthy
            critical_components = ['database', 'exchange', 'position_manager']
            critical_failures = sum(
                1 for c in critical_components
                if c in self.components and not self.components[c].healthy
            )
            
            if critical_failures >= 2:
                return True
                
            # Check error rates
            high_error_components = sum(
                1 for c in self.components.values()
                if c.error_count >= self.thresholds['max_errors']
            )
            
            if high_error_components >= 2:
                return True
                
            # Check system resources
            sys_metrics = self.status.system_metrics
            if any([
                sys_metrics['memory_used_pct'] > 95,
                sys_metrics['disk_used_pct'] > 95,
                sys_metrics['cpu_used_pct'] > 95
            ]):
                return True
                
            return False
            
        except Exception as e:
            handle_error(e, "HealthMonitor.should_emergency_shutdown", logger=self.logger)
            return True  # Fail safe

    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        try:
            now = time.time()
            warnings = []
            errors = []
            
            # Check component health
            for name, comp in self.components.items():
                if not comp.healthy:
                    errors.append(f"{name} unhealthy: {comp.last_error}")
                elif comp.degraded:
                    warnings.append(f"{name} performance degraded")
                    
            # Check system resources
            sys_metrics = self.check_system_resources()
            
            return HealthStatus(
                timestamp=now,
                is_healthy=len(errors) == 0,
                warnings=warnings,
                errors=errors,
                components=self.components.copy(),
                system_metrics=sys_metrics
            )
            
        except Exception as e:
            handle_error(e, "HealthMonitor.get_health_status", logger=self.logger)
            return HealthStatus(
                timestamp=time.time(),
                is_healthy=False,
                warnings=[],
                errors=[str(e)],
                components={},
                system_metrics={}
            )

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            now = time.time()
            status = self.get_health_status()
            
            return {
                'timestamp': datetime.fromtimestamp(now).isoformat(),
                'healthy': status.is_healthy,
                'components': {
                    name: {
                        'healthy': c.healthy,
                        'last_check_age': now - c.last_check,
                        'latency': c.latency,
                        'error_count': c.error_count,
                        'last_error': c.last_error,
                        'degraded': c.degraded
                    }
                    for name, c in status.components.items()
                },
                'system_metrics': status.system_metrics,
                'warnings': status.warnings,
                'errors': status.errors
            }
            
        except Exception as e:
            handle_error(e, "HealthMonitor.get_health_report", logger=self.logger)
            return {
                'timestamp': datetime.now().isoformat(),
                'healthy': False,
                'error': str(e)
            }

    async def update_component_metrics(self, component: str, latency: float):
        """Thread-safe component metric updates"""
        async with self._metric_lock:
            if component not in self.latency_history:
                self.latency_history[component] = deque(maxlen=self.max_history)
            self.latency_history[component].append(latency)

class SystemHealth:
    def __init__(
        self,
        logger: logging.Logger,
        check_interval: int = 60,
        alert_threshold: int = 90
    ):
        self.logger = logger
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self._running = False
        self._health_data: Dict[str, Any] = {}
        self._last_check: Optional[datetime] = None
        
    async def start_monitoring(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                await asyncio.sleep(5)
    
    def stop_monitoring(self) -> None:
        self._running = False
    
    async def _check_system_health(self) -> None:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self._health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'alert': cpu_percent > self.alert_threshold
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'alert': memory.percent > self.alert_threshold
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'percent': disk.percent,
                'alert': disk.percent > self.alert_threshold
            },
            'process': self._get_process_info()
        }
        
        self._last_check = datetime.utcnow()
        
        if any(item.get('alert', False) for item in self._health_data.values()):
            await self._handle_alerts()
    
    def _get_process_info(self) -> Dict[str, Any]:
        process = psutil.Process()
        return {
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'connections': len(process.connections())
        }
    
    async def _handle_alerts(self) -> None:
        self.logger.warning(
            f"System health alerts detected: {json.dumps(self._health_data, indent=2)}"
        )