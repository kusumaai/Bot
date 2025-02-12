from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
import asyncio
from collections import defaultdict
from utils.data_validator import DataValidator
from utils.numeric_handler import NumericHandler
from utils.exceptions import OrderError, InvalidOrderError

class OrderManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.validator = DataValidator(self.logger)
        self.nh = NumericHandler()
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: Union[str, float, Decimal],
        price: Optional[Union[str, float, Decimal]] = None
    ) -> Optional[Dict[str, Any]]:
        valid, error = self.validator.validate_order_params(symbol, side, amount, price)
        if not valid:
            self.logger.error(f"Invalid order parameters: {error}")
            return None

        async with self._lock:
            try:
                amount_decimal = self.nh.to_decimal(amount)
                price_decimal = self.nh.to_decimal(price) if price else None

                order = await self.ctx.exchange_interface.create_order(
                    symbol=symbol,
                    side=side,
                    amount=amount_decimal,
                    price=price_decimal
                )
                
                if order:
                    order_id = order.get('id')
                    if not order_id:
                        raise OrderError("Order response missing 'id'.")
                    self.open_orders[order_id] = order
                    self.order_history[symbol].append(order)
                return order

            except (OrderError, InvalidOrderError) as e:
                self.logger.error(f"Order placement failed: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error in place_order: {e}")
                return None

    async def cancel_order(self, order_id: str) -> bool:
        async with self._lock:
            try:
                order = await self.ctx.exchange_interface.close_order(order_id)
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
            order = await self.ctx.exchange_interface.get_order_status(order_id)
            return order
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            return None 