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
from utils.error_handler import handle_error

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
        self.latency_history: Dict[str, List[float]] = {}
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
        asyncio.create_task(self.monitor_loop())

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
                self.latency_history[component] = []
            self.latency_history[component].append(latency)
            
            # Keep last 1000 measurements
            self.latency_history[component] = self.latency_history[component][-1000:]
            
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

    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Check database
                db_healthy, db_latency, db_error = await self.check_database()
                self.update_component('database', db_healthy, db_latency, db_error)

                # Check exchange if live trading
                if not self.ctx.config.get('paper_mode', True):
                    ex_healthy, ex_latency, ex_error = await self.check_exchange()
                    self.update_component('exchange', ex_healthy, ex_latency, ex_error)

                # Check system resources
                sys_metrics = self.check_system_resources()
                sys_healthy = all([
                    sys_metrics['memory_healthy'],
                    sys_metrics['disk_healthy'],
                    sys_metrics['cpu_healthy']
                ])
                self.update_component('system', sys_healthy, 0)

                # Update overall health status
                self.status = self.get_health_status()
                
                if not self.status.is_healthy:
                    self.logger.warning(
                        f"System health check failed:\n{json.dumps(self.get_health_report(), indent=2)}"
                    )
                    
                    # Check if emergency shutdown needed
                    if self.should_emergency_shutdown():
                        self.logger.critical("Health check triggering emergency shutdown")
                        await self.ctx.circuit_breaker.emergency_shutdown("Critical health check failure")
                        break

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                handle_error(e, "HealthMonitor.monitor_loop", logger=self.logger)
                await asyncio.sleep(5)  # Back off on error

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