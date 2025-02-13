#!/usr/bin/env python3
"""
Module: monitoring/position_monitor.py
Position monitoring with alerts and risk management
"""

import asyncio
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque

from utils.error_handler import handle_error, handle_error_async
from risk.manager import RiskManager
from risk.position import Position
from risk.limits import RiskLimits

@dataclass
class PositionAlert:
    # Required fields (no defaults)
    symbol: str
    alert_type: str
    message: str
    timestamp: float
    severity: str
    
    # Optional fields (with defaults)
    position_size: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    entry_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None

class PositionMonitor:
    def __init__(self, risk_manager: RiskManager, ctx: Any):
        self.risk_manager = risk_manager
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.alerts: deque = deque(maxlen=1000)  # Fixed size alert history
        
        # Get thresholds from config
        monitor_cfg = ctx.config.get("position_monitor", {})
        self.duration_threshold = monitor_cfg.get("duration_threshold", 24 * 3600)  # 24h default
        self.drawdown_threshold = Decimal(str(monitor_cfg.get("drawdown_threshold", "-0.1")))  # 10% default
        self.check_interval = monitor_cfg.get("check_interval", 1)  # 1s default
        self.max_alerts = monitor_cfg.get("max_alerts", 1000)
        self.cleanup_interval = monitor_cfg.get("cleanup_interval", 3600)  # 1h default
        
        self.last_cleanup = time.time()

    async def monitor_positions(self):
        """Monitor positions with proper error handling and alert management"""
        while True:
            try:
                current_time = time.time()
                
                # Cleanup old alerts periodically
                if current_time - self.last_cleanup > self.cleanup_interval:
                    self._cleanup_old_alerts()
                    self.last_cleanup = current_time

                for symbol, position in self.risk_manager.portfolio.positions.items():
                    await self._check_position(position, current_time)

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                handle_error(e, "PositionMonitor.monitor_positions", logger=self.logger)
                await asyncio.sleep(5)  # Back off on error

    async def _check_position(self, position: Position, current_time: float):
        """Check individual position for alerts"""
        try:
            # Check duration
            if current_time - position.entry_time > self.duration_threshold:
                await self._create_alert(
                    position,
                    "DURATION",
                    f"Position held > {self.duration_threshold/3600:.1f}h",
                    "WARNING"
                )

            # Check drawdown
            if position.max_adverse_excursion < self.drawdown_threshold:
                await self._create_alert(
                    position,
                    "DRAWDOWN",
                    f"Position MAE > {abs(float(self.drawdown_threshold)*100)}%",
                    "CRITICAL"
                )

            # Check stop loss proximity
            sl_distance = abs(position.current_price - position.stop_loss) / position.current_price
            if sl_distance < Decimal("0.01"):  # Within 1% of stop loss
                await self._create_alert(
                    position,
                    "STOP_LOSS",
                    f"Position near stop loss ({float(sl_distance)*100:.1f}% away)",
                    "WARNING"
                )

            # Check position size
            max_size = self.risk_manager.get_max_position_size(position.symbol)
            if position.size > max_size:
                await self._create_alert(
                    position,
                    "SIZE",
                    f"Position size ({position.size}) exceeds max ({max_size})",
                    "CRITICAL"
                )

        except Exception as e:
            handle_error(e, f"PositionMonitor._check_position: {position.symbol}", logger=self.logger)

    async def _create_alert(self, position: Position, alert_type: str, 
                           message: str, severity: str):
        """Thread-safe alert creation"""
        try:
            async with self._lock:
                alert = PositionAlert(
                    symbol=position.symbol,
                    alert_type=alert_type,
                    message=message,
                    timestamp=time.time(),
                    severity=severity,
                    position_size=position.size,
                    current_price=position.current_price,
                    entry_price=position.entry_price,
                    unrealized_pnl=position.unrealized_pnl
                )
                
                self.alerts.append(alert)
                
                if severity == "CRITICAL":
                    self.logger.critical(f"{alert.symbol} - {alert.message}")
                else:
                    self.logger.warning(f"{alert.symbol} - {alert.message}")

        except Exception as e:
            await handle_error_async(e, "PositionMonitor._create_alert", self.logger)

    def _cleanup_old_alerts(self, max_age: Optional[float] = None):
        """Remove old alerts"""
        try:
            if not max_age:
                max_age = self.cleanup_interval
                
            current_time = time.time()
            self.alerts = deque(
                alert for alert in self.alerts
                if current_time - alert.timestamp < max_age
            )
            
        except Exception as e:
            handle_error(e, "PositionMonitor._cleanup_old_alerts", logger=self.logger) 