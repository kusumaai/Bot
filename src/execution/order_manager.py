#! /usr/bin/env python3
# src/execution/order_manager.py
"""
Module: src.execution
Provides order management.
"""
import asyncio
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from src.database.queries import DatabaseQueries
from src.utils.error_handler import handle_error_async
from src.utils.exceptions import OrderStoreError
from utils.data_validator import DataValidator
from utils.exceptions import InvalidOrderError, OrderError
from utils.numeric_handler import NumericHandler


# order manager class that manages orders and order history by using the exchange interface and db queries
class OrderManager:
    """Order manager class"""

    def __init__(self, exchange_interface, db, logger):
        self.exchange_interface = exchange_interface
        self.db = db
        self.db_queries = DatabaseQueries(self.db)
        self.logger = logger
        self._lock = asyncio.Lock()
        self.validator = DataValidator(self.logger)
        self.nh = NumericHandler()
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # place an order with risk validation
    async def place_order(
        self, symbol: str, side: str, amount: Decimal, order_type: str, price: Decimal
    ) -> bool:
        """Place an order with risk validation."""
        async with self._lock:
            trade_result = await self.exchange_interface.execute_trade(
                symbol, side, amount, order_type, price
            )
            if not trade_result["success"]:
                self.logger.error(
                    f"Failed to place order for {symbol}: {trade_result['error']}"
                )
                return False
            order_id = trade_result.get("order_id")
            if order_id:
                await self.db_queries.store_order(
                    {
                        "id": order_id,
                        "symbol": symbol,
                        "side": side,
                        "amount": str(amount),
                        "price": str(price),
                        "status": "open",
                    }
                )
                return True
            self.logger.error("Order ID missing from trade result.")
            return False

    # cancel an order
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        async with self._lock:
            try:
                order = await self.exchange_interface.close_order(order_id)
                if order and order.get("status") == "CLOSED":
                    self.open_orders.pop(order_id, None)
                    self.logger.info(f"Order {order_id} canceled successfully.")
                    return True
                else:
                    self.logger.warning(f"Failed to cancel order {order_id}.")
                    return False
            except Exception as e:
                self.logger.error(f"Error cancelling order {order_id}: {e}")
                return False

    # get order status
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
        try:
            order = await self.exchange_interface.get_order_status(order_id)
            return order
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    # store order details in the database
    async def store_order(self, order_details: Dict[str, Any]) -> bool:
        """Stores order details in the database."""
        try:
            await self.db_queries.insert_order(order_details)
            return True
        except Exception as e:
            await handle_error_async(e, "OrderManager.store_order", self.logger)
            raise OrderStoreError(f"Error storing order: {e}")
