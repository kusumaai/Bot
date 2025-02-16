#! /usr/bin/env python3
# src/exchanges/paper_exchange.py
"""
Module: src.exchanges
Provides paper exchange implementation.
"""
import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np

from src.trading.exceptions import ExchangeAPIError


class PaperExchange:
    def __init__(self):
        self.positions = {}
        self.balances = {"USDT": Decimal("10000")}  # Default paper balance
        self.orders = {}
        self.config = {
            "risk_limits": {
                "min_position_size": "0.01",
                "max_position_size": "0.5",
                "max_positions": 10,
                "max_leverage": "3",
                "max_drawdown": "0.2",
                "max_daily_loss": "0.03",
                "emergency_stop_pct": "0.05",
                "risk_factor": "0.02",
                "kelly_scaling": "0.5",
                "max_correlation": "0.7",
                "max_sector_exposure": "0.3",
                "max_volatility": "0.4",
                "min_liquidity": "0.0001",
            }
        }
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self.initialized = False

    async def create_order(
        self, symbol: str, type: str, side: str, amount: float, price: float = None
    ) -> Dict[str, Any]:
        """Create a simulated order"""
        try:
            async with self._lock:
                order_id = str(len(self.orders) + 1)
                order = {
                    "id": order_id,
                    "symbol": symbol,
                    "type": type,
                    "side": side,
                    "amount": Decimal(str(amount)),
                    "price": Decimal(str(price)) if price else None,
                    "status": "closed",
                    "timestamp": int(time.time() * 1000),
                }
                self.orders[order_id] = order

                # Update positions and balances
                await self._update_position(order)
                return order
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise ExchangeAPIError(f"Failed to create order: {e}")

    async def _update_position(self, order: Dict[str, Any]) -> None:
        """Update positions and balances based on order"""
        try:
            symbol = order["symbol"]
            side = order["side"]
            amount = order["amount"]
            price = order["price"]

            if not price:
                return

            # Calculate value in USDT
            value = amount * price

            # Update balances
            if side == "buy":
                if "USDT" in self.balances and self.balances["USDT"] >= value:
                    self.balances["USDT"] -= value
                    self.positions[symbol] = (
                        self.positions.get(symbol, Decimal("0")) + amount
                    )
                else:
                    raise ExchangeAPIError("Insufficient balance for order")
            else:  # sell
                if symbol in self.positions and self.positions[symbol] >= amount:
                    self.positions[symbol] -= amount
                    self.balances["USDT"] = (
                        self.balances.get("USDT", Decimal("0")) + value
                    )
                else:
                    raise ExchangeAPIError("Insufficient position for order")

        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            raise ExchangeAPIError(f"Failed to update position: {e}")

    async def fetch_balance(self) -> Dict[str, Decimal]:
        """Fetch current balances"""
        return self.balances

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """Fetch current positions"""
        return [
            {"symbol": symbol, "amount": amount, "timestamp": int(time.time() * 1000)}
            for symbol, amount in self.positions.items()
            if amount > 0
        ]

    async def close_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Close an existing order"""
        async with self._lock:
            order = self.orders.pop(order_id, None)
            if order:
                order["status"] = "CLOSED"
                order["close_timestamp"] = int(time.time() * 1000)
                return order
            return None

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an order"""
        return self.orders.get(order_id)

    async def fetch_markets(self) -> Dict[str, Any]:
        """Fetch available markets"""
        markets = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "base": "BTC",
                "quote": "USDT",
                "active": True,
                "precision": {"amount": 8, "price": 2},
                "limits": {
                    "amount": {"min": Decimal("0.0001"), "max": Decimal("1000")},
                    "price": {"min": Decimal("1"), "max": Decimal("1000000")},
                    "cost": {"min": Decimal("10")},
                },
            },
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "base": "ETH",
                "quote": "USDT",
                "active": True,
                "precision": {"amount": 8, "price": 2},
                "limits": {
                    "amount": {"min": Decimal("0.001"), "max": Decimal("10000")},
                    "price": {"min": Decimal("1"), "max": Decimal("100000")},
                    "cost": {"min": Decimal("10")},
                },
            },
        }
        return markets

    async def load_markets(self) -> Dict[str, Any]:
        """Load market information"""
        return await self.fetch_markets()

    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch historical candle data"""
        try:
            candles = []
            current_time = int(time.time())
            base_price = Decimal("50000")  # Base price for simulation

            for i in range(limit):
                timestamp = current_time - i * 60
                open_price = base_price + Decimal(str(np.random.normal(0, 100)))
                high_price = open_price + Decimal(str(abs(np.random.normal(0, 50))))
                low_price = open_price - Decimal(str(abs(np.random.normal(0, 50))))
                close_price = (high_price + low_price) / 2
                volume = Decimal(str(abs(np.random.normal(100, 20))))

                candle = {
                    "timestamp": timestamp * 1000,  # Convert to milliseconds
                    "open": str(open_price),
                    "high": str(high_price),
                    "low": str(low_price),
                    "close": str(close_price),
                    "volume": str(volume),
                }
                candles.append(candle)
            return candles

        except Exception as e:
            self.logger.error(f"Error fetching candles: {e}")
            return None

    async def ping(self) -> bool:
        """Test exchange connectivity"""
        await asyncio.sleep(0.1)  # Simulate network delay
        return True

    async def close(self) -> None:
        """Close exchange connection"""
        self.initialized = False
