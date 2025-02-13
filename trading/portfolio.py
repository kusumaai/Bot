#!/usr/bin/env python3
"""
Module: trading/portfolio.py
Portfolio management with proper risk tracking
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
import time
import threading
from datetime import datetime, timedelta
import asyncio
import logging
from collections import deque

from utils.error_handler import handle_error_async
from trading.position import Position
from risk.limits import RiskLimits, load_risk_limits_from_config
from utils.numeric_handler import NumericHandler
from utils.exceptions import PortfolioError

@dataclass
class PortfolioStats:
    """Portfolio performance statistics"""
    total_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    drawdown: Decimal
    peak_value: Decimal
    daily_pnl: Decimal
    total_exposure: Decimal
    leverage: Decimal
    position_count: int

class PortfolioManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.positions: Dict[str, Position] = {}
        self.balance = Decimal(str(ctx.config.get("initial_balance", "10000")))
        self.initialized = False

        # Get risk limits from config with defaults
        risk_config = ctx.config.get("risk_limits", {})
        self.risk_limits = RiskLimits(
            min_position_size=Decimal(str(risk_config.get("min_position_size", "0.01"))),
            max_position_size=Decimal(str(risk_config.get("max_position_size", "0.5"))),
            max_positions=int(risk_config.get("max_positions", 10)),
            max_leverage=Decimal(str(risk_config.get("max_leverage", "3"))),
            max_drawdown=Decimal(str(risk_config.get("max_drawdown", "0.2"))),
            max_daily_loss=Decimal(str(risk_config.get("max_daily_loss", "0.03"))),
            emergency_stop_pct=Decimal(str(risk_config.get("emergency_stop_pct", "0.15"))),
            risk_factor=Decimal(str(risk_config.get("risk_factor", "0.02"))),
            kelly_scaling=Decimal(str(risk_config.get("kelly_scaling", "0.5"))),
            max_correlation=Decimal(str(risk_config.get("max_correlation", "0.7"))),
            max_sector_exposure=Decimal(str(risk_config.get("max_sector_exposure", "0.3"))),
            max_volatility=Decimal(str(risk_config.get("max_volatility", "0.4")))
        )

        # Validate risk limits immediately
        validation_result = self.risk_limits.validate()
        if not validation_result.is_valid:
            raise ValueError(validation_result.error_message)

        self.stats: Optional[PortfolioStats] = None
        self.nh = NumericHandler()

    async def initialize(self) -> bool:
        """Initialize portfolio manager"""
        try:
            if self.initialized:
                return True
            self.risk_limits = load_risk_limits_from_config(self.ctx.config)
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize portfolio manager: {e}")
            return False

    async def update_position(self, symbol: str, position: Position) -> bool:
        """Update position in portfolio"""
        try:
            self.positions[symbol] = position
            await self.update_stats()
            self.logger.info(f"Updated position: {position}")
            return True
        except Exception as e:
            await handle_error_async(e, "PortfolioManager.update_position", self.logger)
            return False

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)

    async def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())

    async def add_position(self, position: Position):
        """Add a new position to the portfolio"""
        try:
            self.positions[position.symbol] = position
            await self.update_stats()
            self.logger.info(f"Added position: {position}")
        except Exception as e:
            await handle_error_async(e, "PortfolioManager.add_position", self.logger)
            raise PortfolioError(f"Failed to add position: {e}") from e

    async def remove_position(self, symbol: str):
        """Remove a position from the portfolio"""
        try:
            if symbol in self.positions:
                del self.positions[symbol]
                await self.update_stats()
                self.logger.info(f"Removed position: {symbol}")
            else:
                self.logger.warning(f"Tried to remove nonexistent position: {symbol}")
        except Exception as e:
            await handle_error_async(e, "PortfolioManager.remove_position", self.logger)

    async def update_stats(self):
        """Update portfolio statistics"""
        try:
            total_value = sum(pos.current_price * pos.size for pos in self.positions.values())
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            # Example drawdown calculation
            drawdown = Decimal('0')  # Implement actual drawdown logic
            peak_value = Decimal('0')  # Implement peak value tracking
            daily_pnl = Decimal('0')  # Implement daily PnL tracking
            total_exposure = sum(pos.size for pos in self.positions.values())
            leverage = (total_value / self.risk_limits.max_position_size) if self.risk_limits.max_position_size > Decimal('0') else Decimal('0')
            position_count = len(self.positions)
            
            self.stats = PortfolioStats(
                total_value=total_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                drawdown=drawdown,
                peak_value=peak_value,
                daily_pnl=daily_pnl,
                total_exposure=total_exposure,
                leverage=leverage,
                position_count=position_count
            )
            self.logger.info(f"Updated portfolio stats: {self.stats}")
        except Exception as e:
            self.logger.error(f"Error calculating portfolio stats: {e}")
            raise PortfolioError(f"Error calculating portfolio stats: {e}") from e 