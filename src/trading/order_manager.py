import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional


class OrderState:
    """Enum-like class for order states"""

    PENDING = "pending"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderManager:
    def __init__(self, exchange_interface, db_queries, logger=None):
        self._lock = asyncio.Lock()
        self.exchange_interface = exchange_interface
        self.db_queries = db_queries
        self.logger = logger or logging.getLogger(__name__)
        self._pending_confirmations = {}
        self._confirmation_timeout = 30  # seconds
        self._max_confirmation_retries = 3
        self._reconciliation_interval = 60  # seconds
        self._last_reconciliation = 0
        self._state_transitions = {}

    async def place_order(
        self, symbol: str, side: str, amount: Decimal, order_type: str, price: Decimal
    ) -> Dict[str, Any]:
        """
        Place an order with proper state management and confirmation.

        Returns:
            Dict containing order status and details
        """
        try:
            async with self._lock:
                # Generate local order ID for tracking
                local_order_id = f"LOCAL_{int(time.time()*1000)}_{symbol}_{side}"

                # Store initial pending state
                await self._store_order_state(
                    local_order_id,
                    OrderState.PENDING,
                    {
                        "symbol": symbol,
                        "side": side,
                        "amount": str(amount),
                        "type": order_type,
                        "price": str(price),
                        "timestamp": int(time.time() * 1000),
                    },
                )

                try:
                    # Execute trade on exchange
                    trade_result = await self.exchange_interface.execute_trade(
                        symbol=symbol,
                        side=side,
                        amount=float(amount),
                        order_type=order_type,
                        price=float(price),
                    )

                    if not trade_result["success"]:
                        await self._handle_failed_order(
                            local_order_id, trade_result.get("error", "Unknown error")
                        )
                        return {
                            "success": False,
                            "error": trade_result.get(
                                "error", "Trade execution failed"
                            ),
                            "local_order_id": local_order_id,
                        }

                    exchange_order_id = trade_result.get("order_id")
                    if not exchange_order_id:
                        await self._handle_failed_order(
                            local_order_id, "No order ID received from exchange"
                        )
                        return {
                            "success": False,
                            "error": "No order ID received from exchange",
                            "local_order_id": local_order_id,
                        }

                    # Update state to confirming
                    await self._store_order_state(
                        local_order_id,
                        OrderState.CONFIRMING,
                        {"exchange_order_id": exchange_order_id, **trade_result},
                    )

                    # Confirm order state with exchange
                    confirmation = await self._confirm_order_state(
                        exchange_order_id, symbol
                    )
                    if not confirmation["success"]:
                        await self._handle_unconfirmed_order(
                            local_order_id, exchange_order_id, confirmation.get("error")
                        )
                        return {
                            "success": False,
                            "error": "Order state confirmation failed",
                            "local_order_id": local_order_id,
                            "exchange_order_id": exchange_order_id,
                            "needs_reconciliation": True,
                        }

                    # Store confirmed order state
                    await self._store_order_state(
                        local_order_id,
                        OrderState.CONFIRMED,
                        {
                            "exchange_order_id": exchange_order_id,
                            "final_state": confirmation["order_state"],
                            "filled_amount": confirmation.get("filled_amount"),
                            "average_price": confirmation.get("average_price"),
                        },
                    )

                    return {
                        "success": True,
                        "local_order_id": local_order_id,
                        "exchange_order_id": exchange_order_id,
                        "order_state": confirmation["order_state"],
                    }

                except Exception as e:
                    await self._handle_failed_order(local_order_id, str(e))
                    raise

        except Exception as e:
            self.logger.error(f"Order placement error: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Order placement failed: {str(e)}"}

    async def _store_order_state(
        self, order_id: str, state: str, details: Dict[str, Any]
    ) -> bool:
        """Store order state in database with transaction safety."""
        try:
            async with self.db_queries.transaction() as conn:
                # Store state transition
                await conn.execute(
                    """
                    INSERT INTO order_state_transitions 
                    (order_id, state, details, timestamp) 
                    VALUES (?, ?, ?, ?)
                    """,
                    [order_id, state, json.dumps(details), int(time.time() * 1000)],
                )

                # Update current state
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO orders 
                    (order_id, current_state, details, last_update)
                    VALUES (?, ?, ?, ?)
                    """,
                    [order_id, state, json.dumps(details), int(time.time() * 1000)],
                )

                self._state_transitions[order_id] = state
                return True

        except Exception as e:
            self.logger.error(f"Failed to store order state: {str(e)}", exc_info=True)
            return False

    async def _confirm_order_state(
        self, exchange_order_id: str, symbol: str
    ) -> Dict[str, Any]:
        """Confirm order state with exchange with retries."""
        for attempt in range(self._max_confirmation_retries):
            try:
                order_info = await self.exchange_interface.get_order(
                    symbol, exchange_order_id
                )
                if order_info and "status" in order_info:
                    return {
                        "success": True,
                        "order_state": order_info["status"],
                        "filled_amount": order_info.get("filled"),
                        "average_price": order_info.get("average"),
                        "remaining": order_info.get("remaining"),
                    }

                await asyncio.sleep(1)  # Backoff between retries

            except Exception as e:
                self.logger.warning(
                    f"Order confirmation attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == self._max_confirmation_retries - 1:
                    return {
                        "success": False,
                        "error": f"Failed to confirm order state after {self._max_confirmation_retries} attempts",
                    }

    async def _handle_failed_order(self, order_id: str, error: str) -> None:
        """Handle failed order state."""
        await self._store_order_state(
            order_id,
            OrderState.FAILED,
            {"error": error, "timestamp": int(time.time() * 1000)},
        )
        self.logger.error(f"Order {order_id} failed: {error}")

    async def _handle_unconfirmed_order(
        self, local_order_id: str, exchange_order_id: str, error: str
    ) -> None:
        """Handle unconfirmed order state and schedule reconciliation."""
        await self._store_order_state(
            local_order_id,
            OrderState.CONFIRMING,
            {
                "error": error,
                "exchange_order_id": exchange_order_id,
                "needs_reconciliation": True,
                "timestamp": int(time.time() * 1000),
            },
        )
        self._pending_confirmations[local_order_id] = {
            "exchange_order_id": exchange_order_id,
            "timestamp": time.time(),
        }
        self.logger.warning(
            f"Order {local_order_id} (exchange ID: {exchange_order_id}) needs reconciliation: {error}"
        )

    async def reconcile_orders(self) -> None:
        """Reconcile pending and unconfirmed orders."""
        try:
            current_time = time.time()
            if current_time - self._last_reconciliation < self._reconciliation_interval:
                return

            self._last_reconciliation = current_time

            async with self.db_queries.transaction() as conn:
                # Get all unconfirmed orders
                unconfirmed_orders = await conn.execute(
                    """
                    SELECT order_id, details 
                    FROM orders 
                    WHERE current_state IN (?, ?)
                    """,
                    [OrderState.PENDING, OrderState.CONFIRMING],
                )

                for order in unconfirmed_orders:
                    details = json.loads(order["details"])
                    exchange_order_id = details.get("exchange_order_id")

                    if not exchange_order_id:
                        # Local-only order that never reached exchange
                        if (
                            current_time - details["timestamp"] / 1000
                            > self._confirmation_timeout
                        ):
                            await self._handle_failed_order(
                                order["order_id"],
                                "Order timed out without reaching exchange",
                            )
                        continue

                    # Check exchange state
                    confirmation = await self._confirm_order_state(
                        exchange_order_id, details["symbol"]
                    )
                    if confirmation["success"]:
                        await self._store_order_state(
                            order["order_id"],
                            OrderState.CONFIRMED,
                            {
                                "exchange_order_id": exchange_order_id,
                                "final_state": confirmation["order_state"],
                                "filled_amount": confirmation.get("filled_amount"),
                                "average_price": confirmation.get("average_price"),
                                "reconciliation_time": int(current_time * 1000),
                            },
                        )
                    elif (
                        current_time - details["timestamp"] / 1000
                        > self._confirmation_timeout
                    ):
                        await self._handle_failed_order(
                            order["order_id"],
                            f"Failed to confirm order state after timeout: {confirmation.get('error', 'Unknown error')}",
                        )

        except Exception as e:
            self.logger.error(f"Order reconciliation error: {str(e)}", exc_info=True)

    async def get_order_state(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current order state and details."""
        try:
            async with self.db_queries.transaction() as conn:
                order = await conn.execute_one(
                    "SELECT current_state, details FROM orders WHERE order_id = ?",
                    [order_id],
                )
                if order:
                    return {
                        "state": order["current_state"],
                        "details": json.loads(order["details"]),
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting order state: {str(e)}", exc_info=True)
            return None
