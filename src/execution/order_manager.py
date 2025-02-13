from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
import asyncio
from collections import defaultdict
from utils.data_validator import DataValidator
from utils.numeric_handler import NumericHandler
from utils.exceptions import OrderError, InvalidOrderError

class OrderManager:
    def __init__(self, exchange_interface, db_queries, logger):
        self.exchange_interface = exchange_interface
        self.db_queries = db_queries
        self.logger = logger
        self._lock = asyncio.Lock()
        self.validator = DataValidator(self.logger)
        self.nh = NumericHandler()
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    async def place_order(self, symbol: str, side: str, amount: Decimal, order_type: str, price: Decimal) -> bool:
        """Place an order with risk validation."""
        async with self._lock:
            trade_result = await self.exchange_interface.execute_trade(symbol, side, amount, order_type, price)
            if not trade_result['success']:
                self.logger.error(f"Failed to place order for {symbol}: {trade_result['error']}")
                return False
            order_id = trade_result.get('order_id')
            if order_id:
                await self.db_queries.store_order({'id': order_id, 'symbol': symbol, 'side': side, 'amount': str(amount), 'price': str(price), 'status': 'open'})
                return True
            self.logger.error("Order ID missing from trade result.")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        async with self._lock:
            try:
                order = await self.exchange_interface.close_order(order_id)
                if order and order.get('status') == 'CLOSED':
                    self.open_orders.pop(order_id, None)
                    self.logger.info(f"Order {order_id} canceled successfully.")
                    return True
                else:
                    self.logger.warning(f"Failed to cancel order {order_id}.")
                    return False
            except Exception as e:
                self.logger.error(f"Error cancelling order {order_id}: {e}")
                return False

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            order = await self.exchange_interface.get_order_status(order_id)
            return order
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            return None 