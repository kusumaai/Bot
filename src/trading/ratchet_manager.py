import asyncio
import time
from decimal import Decimal, InvalidOperation
from typing import Any


class RatchetManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self._active_trades = {}
        self._last_update = {}
        self._update_interval = 5  # seconds
        self._max_iterations = 10  # Maximum iterations for stop adjustment
        self._min_price_change = Decimal(
            "0.0001"
        )  # Minimum price change to trigger update
        self._max_update_time = 30  # Maximum time for a single update in seconds
        self._last_health_check = time.time()
        self._health_check_interval = 60  # seconds
        self._failed_updates = {}
        self._max_failed_attempts = 3

    async def _update_single_trailing_stop(
        self, trade_id: str, trade_data: dict, current_price: Decimal
    ) -> bool:
        """
        Update trailing stop for a single trade with safety controls.

        Args:
            trade_id: Unique trade identifier
            trade_data: Trade data including current stop
            current_price: Current market price

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            start_time = time.time()
            iterations = 0
            last_stop = Decimal(str(trade_data.get("current_stop", "0")))
            initial_price = current_price

            while iterations < self._max_iterations:
                # Check timeout
                if time.time() - start_time > self._max_update_time:
                    self.logger.error(
                        f"Trailing stop update timeout for trade {trade_id}",
                        extra={
                            "trade_id": trade_id,
                            "iterations": iterations,
                            "elapsed_time": time.time() - start_time,
                        },
                    )
                    await self._handle_update_failure(trade_id, "Update timeout")
                    return False

                # Calculate new stop price
                try:
                    stop_price = await self._calculate_new_stop(
                        trade_data, current_price
                    )
                except Exception as e:
                    self.logger.error(
                        f"Stop calculation error for trade {trade_id}: {e}"
                    )
                    await self._handle_update_failure(
                        trade_id, f"Calculation error: {e}"
                    )
                    return False

                # Validate stop price
                if not self._validate_stop_price(stop_price, current_price, trade_data):
                    self.logger.error(
                        f"Invalid stop price calculated for trade {trade_id}",
                        extra={
                            "stop_price": str(stop_price),
                            "current_price": str(current_price),
                        },
                    )
                    await self._handle_update_failure(trade_id, "Invalid stop price")
                    return False

                # Check if update is needed
                price_change = abs(stop_price - last_stop)
                if price_change < self._min_price_change:
                    break

                # Verify price hasn't moved significantly during update
                current_price = await self._get_current_price(trade_data["symbol"])
                if abs(current_price - initial_price) / initial_price > Decimal("0.01"):
                    self.logger.warning(
                        f"Price moved significantly during update for trade {trade_id}",
                        extra={
                            "initial_price": str(initial_price),
                            "current_price": str(current_price),
                        },
                    )
                    return False

                # Update stop with transaction safety
                success = await self._safe_update_stop(trade_id, stop_price, trade_data)
                if not success:
                    return False

                last_stop = stop_price
                iterations += 1

                # Small delay between iterations
                await asyncio.sleep(0.1)

            if iterations == self._max_iterations:
                self.logger.warning(
                    f"Max iterations reached for trade {trade_id}",
                    extra={"iterations": iterations},
                )
                return False

            # Reset failed attempts on successful update
            if trade_id in self._failed_updates:
                del self._failed_updates[trade_id]

            self._last_update[trade_id] = time.time()
            return True

        except Exception as e:
            self.logger.error(
                f"Error updating trailing stop for trade {trade_id}: {e}", exc_info=True
            )
            await self._handle_update_failure(trade_id, str(e))
            return False

    async def _safe_update_stop(
        self, trade_id: str, stop_price: Decimal, trade_data: dict
    ) -> bool:
        """Safely update stop price with transaction protection."""
        try:
            async with self.ctx.db_connection.transaction() as conn:
                # Verify trade still exists and state hasn't changed
                current_trade = await conn.execute_one(
                    "SELECT state FROM trades WHERE id = ?", [trade_id]
                )
                if not current_trade or current_trade["state"] != trade_data["state"]:
                    self.logger.error(f"Trade state changed during update: {trade_id}")
                    return False

                # Update stop price
                await conn.execute(
                    """
                    UPDATE trades 
                    SET stop_price = ?, 
                        last_update = ?,
                        update_count = update_count + 1
                    WHERE id = ?
                    """,
                    [str(stop_price), int(time.time() * 1000), trade_id],
                )

                # Store update history
                await conn.execute(
                    """
                    INSERT INTO stop_updates 
                    (trade_id, old_stop, new_stop, update_time)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        trade_id,
                        str(trade_data["current_stop"]),
                        str(stop_price),
                        int(time.time() * 1000),
                    ],
                )

                trade_data["current_stop"] = stop_price
                return True

        except Exception as e:
            self.logger.error(f"Failed to update stop price: {e}", exc_info=True)
            return False

    async def _handle_update_failure(self, trade_id: str, reason: str) -> None:
        """Handle failed stop update with retry tracking."""
        self._failed_updates[trade_id] = self._failed_updates.get(trade_id, 0) + 1

        if self._failed_updates[trade_id] >= self._max_failed_attempts:
            self.logger.error(
                f"Max failed attempts reached for trade {trade_id}. Emergency closure required.",
                extra={
                    "trade_id": trade_id,
                    "failed_attempts": self._failed_updates[trade_id],
                    "reason": reason,
                },
            )
            await self._trigger_emergency_closure(trade_id)

    def _validate_stop_price(
        self, stop_price: Decimal, current_price: Decimal, trade_data: dict
    ) -> bool:
        """Validate calculated stop price."""
        try:
            if stop_price <= 0:
                return False

            # Validate against current price
            min_distance = current_price * Decimal("0.001")  # 0.1% minimum distance
            max_distance = current_price * Decimal("0.1")  # 10% maximum distance

            price_distance = abs(current_price - stop_price)
            if price_distance < min_distance or price_distance > max_distance:
                return False

            # Validate against original entry
            entry_price = Decimal(str(trade_data["entry_price"]))
            if trade_data["side"] == "buy":
                if stop_price > current_price or stop_price < entry_price * Decimal(
                    "0.5"
                ):
                    return False
            else:  # sell
                if stop_price < current_price or stop_price > entry_price * Decimal(
                    "1.5"
                ):
                    return False

            return True

        except (TypeError, InvalidOperation) as e:
            self.logger.error(f"Stop price validation error: {e}")
            return False

    async def _trigger_emergency_closure(self, trade_id: str) -> None:
        """Trigger emergency closure of a trade."""
        try:
            self.logger.error(f"Initiating emergency closure for trade {trade_id}")
            # Notify risk manager
            if hasattr(self.ctx, "risk_manager"):
                await self.ctx.risk_manager.handle_emergency_closure(trade_id)
            # Notify circuit breaker
            if hasattr(self.ctx, "circuit_breaker"):
                await self.ctx.circuit_breaker.trigger_emergency_stop(
                    f"Trailing stop update failure for trade {trade_id}"
                )
        except Exception as e:
            self.logger.error(f"Emergency closure failed: {e}", exc_info=True)
