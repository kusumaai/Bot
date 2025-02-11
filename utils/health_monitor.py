#!/usr/bin/env python3
"""
Module: utils/health_monitor.py
Production health monitoring system
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class ComponentHealth:
    healthy: bool
    last_check: float
    latency: float
    error_count: int
    last_error: Optional[str]

class HealthMonitor:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.start_time = time.time()
        self.last_heartbeat = self.start_time
        
        # Component tracking
        self.components: Dict[str, ComponentHealth] = {
            'database': ComponentHealth(True, time.time(), 0, 0, None),
            'exchange': ComponentHealth(True, time.time(), 0, 0, None),
            'order_manager': ComponentHealth(True, time.time(), 0, 0, None),
            'position_manager': ComponentHealth(True, time.time(), 0, 0, None),
            'system': ComponentHealth(True, time.time(), 0, 0, None)
        }
        
        # Performance tracking
        self.latency_history: Dict[str, List[float]] = {}
        self.error_window = timedelta(minutes=5)
        
        # System metrics
        self.memory_threshold = ctx.config.get("memory_threshold_pct", 90)
        self.disk_threshold = ctx.config.get("disk_threshold_pct", 90)
        self.cpu_threshold = ctx.config.get("cpu_threshold_pct", 80)
        
        # Degradation tracking
        self.latency_baseline: Dict[str, float] = {}
        self.degradation_threshold = 2.0  # 2x baseline is considered degraded
        
        # Start monitoring
        asyncio.create_task(self.monitor_loop())

    def update_component(
        self,
        component: str,
        healthy: bool,
        latency: float,
        error: Optional[str] = None
    ) -> None:
        """Update component health status"""
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
        
        # Update baseline if we have enough data
        if len(self.latency_history[component]) >= 100:
            self.latency_baseline[component] = np.median(self.latency_history[component])
            
        # Check for degradation
        if component in self.latency_baseline:
            if latency > self.latency_baseline[component] * self.degradation_threshold:
                self.ctx.logger.warning(
                    f"Performance degradation detected for {component}: "
                    f"Current={latency:.3f}s, Baseline={self.latency_baseline[component]:.3f}s"
                )

    async def check_database(self) -> Tuple[bool, float, Optional[str]]:
        """Check database connectivity"""
        start = time.time()
        try:
            with self.ctx.db_pool.connection() as conn:
                conn.execute("SELECT 1")
            return True, time.time() - start, None
        except Exception as e:
            return False, time.time() - start, str(e)

    async def check_exchange(self) -> Tuple[bool, float, Optional[str]]:
        """Check exchange connectivity"""
        start = time.time()
        try:
            await self.ctx.exchange_interface.ping()
            return True, time.time() - start, None
        except Exception as e:
            return False, time.time() - start, str(e)

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory_used_pct': memory.percent,
            'memory_healthy': memory.percent < self.memory_threshold,
            'disk_used_pct': disk.percent,
            'disk_healthy': disk.percent < self.disk_threshold,
            'cpu_used_pct': cpu_percent,
            'cpu_healthy': cpu_percent < self.cpu_threshold
        }

    def check_position_manager(self) -> Tuple[bool, float, Optional[str]]:
        """Check position manager health"""
        start = time.time()
        try:
            status = self.ctx.position_manager.get_portfolio_status()
            if not status:
                return False, time.time() - start, "Failed to get portfolio status"
            return True, time.time() - start, None
        except Exception as e:
            return False, time.time() - start, str(e)

    def check_order_manager(self) -> Tuple[bool, float, Optional[str]]:
        """Check order manager health"""
        start = time.time()
        try:
            # Verify recent orders are being tracked
            recent = self.ctx.order_manager.get_recent_orders()
            if recent is None:
                return False, time.time() - start, "Failed to get recent orders"
            return True, time.time() - start, None
        except Exception as e:
            return False, time.time() - start, str(e)

    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Check database
                db_healthy, db_latency, db_error = await self.check_database()
                self.update_component('database', db_healthy, db_latency, db_error)

                # Check exchange if trading enabled
                if not self.ctx.config.get('paper_mode', True):
                    ex_healthy, ex_latency, ex_error = await self.check_exchange()
                    self.update_component('exchange', ex_healthy, ex_latency, ex_error)

                # Check position manager
                pos_healthy, pos_latency, pos_error = self.check_position_manager()
                self.update_component('position_manager', pos_healthy, pos_latency, pos_error)

                # Check order manager
                ord_healthy, ord_latency, ord_error = self.check_order_manager()
                self.update_component('order_manager', ord_healthy, ord_latency, ord_error)

                # Check system resources
                sys_metrics = self.check_system_resources()
                sys_healthy = all([
                    sys_metrics['memory_healthy'],
                    sys_metrics['disk_healthy'],
                    sys_metrics['cpu_healthy']
                ])
                self.update_component('system', sys_healthy, 0)

                # Update overall health status
                health_report = self.get_health_report()
                
                if not health_report['healthy']:
                    self.ctx.logger.warning(f"System health check failed: {json.dumps(health_report, indent=2)}")
                    
                    # Check if emergency shutdown needed
                    if self.should_emergency_shutdown(health_report):
                        self.ctx.logger.critical("Health check triggering emergency shutdown")
                        await self.ctx.error_handler.execute_emergency_shutdown()
                        break

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.ctx.logger.error(f"Error in health monitor loop: {str(e)}")
                await asyncio.sleep(5)  # Back off on error

    def should_emergency_shutdown(self, health_report: Dict[str, Any]) -> bool:
        """Determine if emergency shutdown needed"""
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
            if c.error_count >= 5  # 5+ errors in window
        )
        
        if high_error_components >= 2:
            return True
            
        # Check system resources
        sys_metrics = health_report['system_metrics']
        if (
            sys_metrics['memory_used_pct'] > 95 or  # Critical memory
            sys_metrics['disk_used_pct'] > 95 or    # Critical disk
            sys_metrics['cpu_used_pct'] > 95        # Critical CPU
        ):
            return True
            
        return False

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        now = time.time()
        
        # Component status
        components = {
            name: {
                'healthy': c.healthy,
                'last_check_age': now - c.last_check,
                'latency': c.latency,
                'error_count': c.error_count,
                'last_error': c.last_error
            }
            for name, c in self.components.items()
        }
        
        # Performance metrics
        performance = {
            name: {
                'current_latency': c.latency,
                'baseline_latency': self.latency_baseline.get(name, 0),
                'degraded': (
                    c.latency > self.latency_baseline.get(name, 0) * self.degradation_threshold
                    if name in self.latency_baseline else False
                )
            }
            for name, c in self.components.items()
        }
        
        # System metrics
        sys_metrics = self.check_system_resources()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime': now - self.start_time,
            'healthy': all(c.healthy for c in self.components.values()),
            'components': components,
            'performance': performance,
            'system_metrics': sys_metrics,
            'error_counts': {
                name: c.error_count
                for name, c in self.components.items()
                if c.error_count > 0
            }
        }

    def log_health_metrics(self) -> None:
        """Log health metrics for monitoring"""
        report = self.get_health_report()
        
        # Log overall status
        self.ctx.logger.info(
            f"Health Status: {'HEALTHY' if report['healthy'] else 'UNHEALTHY'} "
            f"Uptime: {report['uptime']:.1f}s"
        )
        
        # Log unhealthy components
        unhealthy = [
            name for name, c in report['components'].items()
            if not c['healthy']
        ]
        if unhealthy:
            self.ctx.logger.warning(f"Unhealthy components: {', '.join(unhealthy)}")
            
        # Log performance issues
        degraded = [
            name for name, p in report['performance'].items()
            if p['degraded']
        ]
        if degraded:
            self.ctx.logger.warning(f"Degraded components: {', '.join(degraded)}")
            
        # Log resource usage
        sys = report['system_metrics']
        self.ctx.logger.info(
            f"System Resources - Memory: {sys['memory_used_pct']}%, "
            f"Disk: {sys['disk_used_pct']}%, CPU: {sys['cpu_used_pct']}%"
        )