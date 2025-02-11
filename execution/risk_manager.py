#!/usr/bin/env python3
"""
Module: execution/position_manager.py
Enhanced position manager with comprehensive risk management
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.error_handler import handle_error
from trading.math import (
    calculate_kelly_fraction,
    calculate_position_size,
    calculate_expected_value
)

@dataclass
class PositionInfo:
    """Position tracking dataclass"""
    symbol: str
    direction: str
    entry_price: float
    current_price: float
    size: float
    entry_time: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

class PositionManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.positions: Dict[str, PositionInfo] = {}
        self.position_history: List[Dict[str, Any]] = []
        
        # Risk limits
        self.max_positions = ctx.config.get("max_positions", 3)
        self.max_correlation = ctx.config.get("max_correlation", 0.7)
        self.max_leverage = Decimal(str(ctx.config.get("max_leverage", 2.0)))
        self.max_position_size = Decimal(str(ctx.config.get("max_position_size", 0.1)))
        self.max_drawdown = Decimal(str(ctx.config.get("max_drawdown_pct", 10))) / 100
        
        # Portfolio tracking
        self.initial_balance = Decimal(str(ctx.config.get("initial_balance", 10000)))
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        
        # Time tracking
        self.last_update = time.time()
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)

    def _calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value including open positions"""
        portfolio_value = self.current_balance
        
        for pos in self.positions.values():
            pnl = Decimal(str(pos.unrealized_pnl))
            portfolio_value += pnl
            
        return portfolio_value

    def _calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown percentage"""
        portfolio_value = self._calculate_portfolio_value()
        drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
        return drawdown

    def _check_correlation(self, symbol: str, returns: np.ndarray) -> bool:
        """Check correlation with existing positions"""
        if not self.positions:
            return True
            
        for pos in self.positions.values():
            if pos.symbol == symbol:
                continue
                
            # Get historical returns for other position
            other_returns = self._get_symbol_returns(pos.symbol)
            if len(other_returns) > 1 and len(returns) > 1:
                correlation = np.corrcoef(returns, other_returns)[0,1]
                if abs(correlation) > self.max_correlation:
                    self.ctx.logger.warning(
                        f"Correlation {correlation:.2f} exceeds maximum {self.max_correlation} "
                        f"between {symbol} and {pos.symbol}"
                    )
                    return False
                    
        return True

    def _get_symbol_returns(self, symbol: str) -> np.ndarray:
        """Get recent returns for symbol from market data"""
        try:
            with self.ctx.db_pool.connection() as conn:
                query = """
                    SELECT close 
                    FROM candles 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """
                df = pd.read_sql_query(query, conn, params=[symbol])
                if len(df) > 1:
                    returns = np.diff(np.log(df['close']))
                    return returns
                return np.array([])
        except Exception as e:
            handle_error(e, "PositionManager._get_symbol_returns", self.ctx.logger)
            return np.array([])

    def _validate_new_position(
        self, 
        symbol: str,
        size: float,
        current_price: float,
        returns: np.ndarray
    ) -> bool:
        """Validate new position against risk limits"""
        try:
            # Check max positions
            if len(self.positions) >= self.max_positions:
                self.ctx.logger.warning(f"Maximum positions {self.max_positions} reached")
                return False
                
            # Check correlation
            if not self._check_correlation(symbol, returns):
                return False
                
            # Check position size
            position_value = Decimal(str(size * current_price))
            portfolio_value = self._calculate_portfolio_value()
            position_ratio = position_value / portfolio_value
            
            if position_ratio > self.max_position_size:
                self.ctx.logger.warning(
                    f"Position size {position_ratio:.2%} exceeds maximum {self.max_position_size:.2%}"
                )
                return False
                
            # Check leverage
            total_exposure = sum(
                Decimal(str(p.size * p.current_price)) 
                for p in self.positions.values()
            ) + position_value
            
            leverage = total_exposure / portfolio_value
            if leverage > self.max_leverage:
                self.ctx.logger.warning(
                    f"Leverage {leverage:.2f} would exceed maximum {self.max_leverage:.2f}"
                )
                return False
                
            # Check drawdown
            if self._calculate_drawdown() > self.max_drawdown:
                self.ctx.logger.warning(
                    f"Maximum drawdown {self.max_drawdown:.2%} exceeded"
                )
                return False
                
            return True
            
        except Exception as e:
            handle_error(e, "PositionManager._validate_new_position", self.ctx.logger)
            return False

    def calculate_position_params(
        self,
        signal: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate position parameters with comprehensive risk checks"""
        try:
            current_price = market_data["current_price"]
            volatility = market_data["volatility"]
            returns = market_data["returns"]
            
            # Calculate expected value
            ev, win_target, loss_target = calculate_expected_value(
                current_price,
                market_data["predicted_return"],
                signal.get("probability", 0.5),
                self.ctx.config.get("stop_loss_pct", 0.02),
                self.ctx.config.get("transaction_cost", 0.001)
            )
            
            # Skip if negative EV
            if ev <= 0:
                return {"position_size": 0}
                
            # Calculate Kelly fraction
            kelly = calculate_kelly_fraction(
                signal.get("probability", 0.5),
                win_target,
                loss_target
            )
            
            # Apply Kelly scaling
            kelly *= self.ctx.config.get("kelly_scaling", 0.5)
            
            # Calculate base position size
            position_size = calculate_position_size(
                float(self._calculate_portfolio_value()),
                kelly,
                current_price,
                volatility,
                self.ctx.config.get("risk_factor", 0.1)
            )
            
            # Validate position
            if not self._validate_new_position(
                signal["symbol"],
                position_size,
                current_price,
                returns
            ):
                return {"position_size": 0}
                
            return {
                "position_size": position_size,
                "kelly_fraction": kelly,
                "expected_value": ev,
                "stop_loss": current_price * (1 - self.ctx.config.get("stop_loss_pct", 0.02)),
                "take_profit": current_price * (1 + self.ctx.config.get("take_profit_pct", 0.03))
            }
            
        except Exception as e:
            handle_error(e, "PositionManager.calculate_position_params", self.ctx.logger)
            return {"position_size": 0}

    def update_position(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Update position status and check for exit conditions"""
        try:
            if symbol not in self.positions:
                return None
                
            pos = self.positions[symbol]
            pos.current_price = current_price
            
            # Calculate unrealized PnL
            price_diff = current_price - pos.entry_price
            multiplier = 1 if pos.direction == "long" else -1
            pos.unrealized_pnl = price_diff * pos.size * multiplier
            
            # Update peak balance if needed
            portfolio_value = self._calculate_portfolio_value()
            if portfolio_value > self.peak_balance:
                self.peak_balance = portfolio_value
                
            # Check stop loss
            if (pos.direction == "long" and current_price <= pos.stop_loss) or \
               (pos.direction == "short" and current_price >= pos.stop_loss):
                return {
                    "action": "close",
                    "reason": "stop_loss",
                    "position": pos
                }
                
            # Check take profit
            if (pos.direction == "long" and current_price >= pos.take_profit) or \
               (pos.direction == "short" and current_price <= pos.take_profit):
                return {
                    "action": "close",
                    "reason": "take_profit",
                    "position": pos
                }
                
            # Check max hold time
            hold_time = time.time() - pos.entry_time
            max_hold_seconds = self.ctx.config.get("max_hold_hours", 24) * 3600
            if hold_time >= max_hold_seconds:
                return {
                    "action": "close",
                    "reason": "max_hold_time",
                    "position": pos
                }
                
            # Check drawdown
            if self._calculate_drawdown() > self.max_drawdown:
                return {
                    "action": "close",
                    "reason": "max_drawdown",
                    "position": pos
                }
                
            return None
            
        except Exception as e:
            handle_error(e, "PositionManager.update_position", self.ctx.logger)
            return None

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str
    ) -> Optional[Dict[str, Any]]:
        """Close position and update tracking"""
        try:
            if symbol not in self.positions:
                return None
                
            pos = self.positions[symbol]
            
            # Calculate final PnL
            price_diff = exit_price - pos.entry_price
            multiplier = 1 if pos.direction == "long" else -1
            realized_pnl = price_diff * pos.size * multiplier
            
            # Update balance
            self.current_balance += Decimal(str(realized_pnl))
            
            # Record trade
            trade_record = {
                "symbol": symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "size": pos.size,
                "entry_time": pos.entry_time,
                "exit_time": time.time(),
                "pnl": realized_pnl,
                "exit_reason": reason
            }
            self.position_history.append(trade_record)
            
            # Remove position
            del self.positions[symbol]
            
            return trade_record
            
        except Exception as e:
            handle_error(e, "PositionManager.close_position", self.ctx.logger)
            return None

    def add_position(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        """Add new position with validation"""
        try:
            # Final validation
            if symbol in self.positions:
                return False
                
            position = PositionInfo(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                current_price=entry_price,
                size=size,
                entry_time=time.time(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                unrealized_pnl=0.0
            )
            
            self.positions[symbol] = position
            return True
            
        except Exception as e:
            handle_error(e, "PositionManager.add_position", self.ctx.logger)
            return False

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            drawdown = self._calculate_drawdown()
            
            return {
                "portfolio_value": float(portfolio_value),
                "current_balance": float(self.current_balance),
                "peak_balance": float(self.peak_balance),
                "drawdown": float(drawdown),
                "open_positions": len(self.positions),
                "total_exposure": sum(
                    p.size * p.current_price for p in self.positions.values()
                ),
                "daily_pnl": float(self.current_balance - self.daily_start_balance),
                "positions": [
                    {
                        "symbol": s,
                        "direction": p.direction,
                        "size": p.size,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl
                    }
                    for s, p in self.positions.items()
                ]
            }
            
        except Exception as e:
            handle_error(e, "PositionManager.get_portfolio_status", self.ctx.logger)
            return {}

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics"""
        now = datetime.now()
        if now > self.daily_reset_time + timedelta(days=1):
            self.daily_start_balance = self.current_balance
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)