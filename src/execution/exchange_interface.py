#! /usr/bin/env python3
# src/execution/exchange_interface.py
"""
Module: src.execution
Handles all exchange interactions with proper resource management.
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

from src.database.database import DatabaseConnection
from src.exchanges.exchange_manager import ExchangeManager
from src.risk.validation import MarketDataValidation
from src.trading.exceptions import (
    DatabaseError,
    ExchangeError,
    PositionError,
    ValidationError,
)
from src.utils.error_handler import handle_error_async
from src.utils.numeric_handler import NumericHandler


class ExchangeInterface:
    """Manages exchange interactions with proper resource management."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.initialized = False

        # Use preferred exchange from config, default to 'paper'
        self.exchange_id = ctx.config.get("preferred_exchange", "paper")
        self.exchange_manager = self._create_exchange_manager()

        # Ensure exchange manager has config
        if not hasattr(self.exchange_manager, "config"):
            self.exchange_manager.config = ctx.config

        self.risk_manager = ctx.risk_manager
        self.db_queries = ctx.db_queries if hasattr(ctx, "db_queries") else None

    def _create_exchange_manager(self) -> ExchangeManager:
        """Create appropriate exchange manager with proper configuration."""
        if self.exchange_id == "paper":
            from src.exchanges.paper_exchange import PaperExchangeManager

            return PaperExchangeManager(
                self.ctx.config, self.logger, self.ctx.db_queries
            )
        else:
            return ExchangeManager(
                self.exchange_id,
                sandbox=True,
                logger=self.logger,
                db_queries=self.ctx.db_queries,
            )

    async def initialize(self) -> bool:
        """Initialize exchange interface with proper error handling."""
        if self.initialized:
            return True

        async with self._lock:
            try:
                if (
                    self.ctx.config.get("paper_mode", False)
                    and self.exchange_id == "paper"
                ):
                    self.logger.info(
                        "Paper mode enabled - skipping real exchange initialization."
                    )
                    self.initialized = True
                    return True

                if await self.exchange_manager.initialize():
                    self.initialized = True
                    self.logger.info("Exchange interface initialized successfully.")
                    return True

                self.logger.error("Failed to initialize exchange manager.")
                return False

            except Exception as e:
                await handle_error_async(e, "ExchangeInterface.initialize", self.logger)
                return False

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        order_type: str,
        price: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Execute a trade with proper validation and resource management."""
        if not self.initialized and not await self.initialize():
            return {"success": False, "error": "Exchange not initialized"}

        try:
            # Validate trade
            is_valid, error = await self.risk_manager.validate_trade(
                symbol, side, amount, price
            )
            if not is_valid:
                return {"success": False, "error": error}

            # Execute order with proper resource management
            async with self.exchange_manager.rate_limiter.acquire_context("trade"):
                order = await self.exchange_manager.create_order(
                    symbol, side, amount, price, order_type=order_type
                )

                # Log trade if database is available
                if self.db_queries:
                    try:
                        await self.db_queries.store_trade(
                            {
                                "symbol": symbol,
                                "side": side,
                                "amount": str(amount),
                                "price": str(price) if price else None,
                                "order_type": order_type,
                                "order_id": order["id"],
                                "timestamp": datetime.utcnow().timestamp(),
                            }
                        )
                    except DatabaseError as e:
                        self.logger.error(f"Failed to log trade: {e}")
                        # Continue since trade was successful

                return {"success": True, "order_id": order["id"]}

        except ValidationError as e:
            self.logger.error(f"Trade validation failed: {e}")
            return {"success": False, "error": str(e)}
        except ExchangeError as e:
            await handle_error_async(e, "ExchangeInterface.execute_trade", self.logger)
            return {"success": False, "error": f"Exchange error: {str(e)}"}
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.execute_trade", self.logger)
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    async def fetch_ticker(self, symbol: str) -> Optional[float]:
        """Fetch ticker data with proper resource management."""
        if not self.initialized and not await self.initialize():
            return None

        try:
            async with self.exchange_manager.rate_limiter.acquire_context("market"):
                ticker = await self.exchange_manager.fetch_ticker(symbol)

                # Log ticker if database is available
                if self.db_queries:
                    try:
                        await self.db_queries.store_ticker(symbol, ticker)
                    except DatabaseError as e:
                        self.logger.error(f"Failed to log ticker: {e}")
                        # Continue since we have the ticker data

                return ticker.get("last")

        except ExchangeError as e:
            await handle_error_async(e, "ExchangeInterface.fetch_ticker", self.logger)
            return None
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.fetch_ticker", self.logger)
            return None

    async def close_position(self, symbol: str, amount: Decimal) -> Dict[str, Any]:
        """
        Close a position with proper state verification and rollback.

        Args:
            symbol: Trading pair symbol
            amount: Position size to close

        Returns:
            Dict containing operation status and details
        """
        order_id = None
        try:
            # Start transaction for atomic state updates
            async with self.db_queries.transaction() as conn:
                # 1. Verify position exists and is open
                position = await conn.execute_one(
                    """
                    SELECT id, state, amount, entry_price 
                    FROM positions 
                    WHERE symbol = ? AND state = 'OPEN'
                    """,
                    [symbol],
                )

                if not position:
                    raise PositionError(f"No open position found for {symbol}")

                # 2. Update position state to CLOSING
                await conn.execute(
                    """
                    UPDATE positions 
                    SET state = 'CLOSING',
                        update_time = ?,
                        update_count = update_count + 1
                    WHERE id = ? AND state = 'OPEN'
                    """,
                    [int(time.time() * 1000), position["id"]],
                )

                # 3. Execute exchange order with retry and verification
                async with self.exchange_manager.rate_limiter.acquire_context("trade"):
                    order = await self.exchange_manager.create_order(
                        symbol=symbol,
                        side="sell",
                        amount=float(amount),
                        order_type="market",
                    )

                    if not order or "id" not in order:
                        raise ExchangeError("Failed to create close order")

                    order_id = order["id"]

                    # 4. Verify order execution
                    verification_attempts = 0
                    max_attempts = 5
                    verification_delay = 1  # seconds

                    while verification_attempts < max_attempts:
                        try:
                            order_status = await self.exchange_manager.fetch_order(
                                symbol, order_id
                            )
                            if order_status.get("status") == "closed":
                                break
                            if order_status.get("status") in [
                                "canceled",
                                "expired",
                                "rejected",
                            ]:
                                raise ExchangeError(
                                    f"Order {order_id} failed: {order_status.get('status')}"
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"Order verification attempt {verification_attempts + 1} failed: {e}"
                            )

                        verification_attempts += 1
                        await asyncio.sleep(verification_delay)

                    if verification_attempts >= max_attempts:
                        raise ExchangeError(
                            f"Failed to verify order {order_id} execution after {max_attempts} attempts"
                        )

                    # 5. Verify position closure on exchange
                    exchange_positions = await self.exchange_manager.fetch_positions(
                        [symbol]
                    )
                    if any(
                        p.get("symbol") == symbol and float(p.get("size", 0)) > 0
                        for p in exchange_positions
                    ):
                        raise PositionError(
                            f"Position {symbol} still open on exchange after closure attempt"
                        )

                    # 6. Update local state after successful verification
                    close_price = Decimal(
                        str(order_status.get("average", order_status.get("price", 0)))
                    )
                    if close_price <= 0:
                        raise ValueError("Invalid close price")

                    realized_pnl = (
                        close_price - Decimal(str(position["entry_price"]))
                    ) * amount

                    await conn.execute(
                        """
                        UPDATE positions 
                        SET state = 'CLOSED',
                            close_price = ?,
                            close_time = ?,
                            realized_pnl = ?,
                            close_order_id = ?,
                            update_time = ?,
                            metadata = json_set(
                                COALESCE(metadata, '{}'),
                                '$.close_order_details',
                                ?
                            )
                        WHERE id = ? AND state = 'CLOSING'
                        """,
                        [
                            str(close_price),
                            int(time.time() * 1000),
                            str(realized_pnl),
                            order_id,
                            int(time.time() * 1000),
                            json.dumps(order_status),
                            position["id"],
                        ],
                    )

                    # 7. Log closure in trade history
                    await conn.execute(
                        """
                        INSERT INTO trade_history 
                        (position_id, action, price, amount, timestamp, metadata)
                        VALUES (?, 'CLOSE', ?, ?, ?, ?)
                        """,
                        [
                            position["id"],
                            str(close_price),
                            str(amount),
                            int(time.time() * 1000),
                            json.dumps(
                                {
                                    "order_id": order_id,
                                    "execution_details": order_status,
                                }
                            ),
                        ],
                    )

                    return {
                        "success": True,
                        "position_id": position["id"],
                        "order_id": order_id,
                        "close_price": str(close_price),
                        "realized_pnl": str(realized_pnl),
                    }

        except Exception as e:
            error_msg = f"Position closure failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Attempt to rollback if order was placed but verification failed
            if order_id:
                try:
                    await self._handle_failed_closure(symbol, order_id, str(e))
                except Exception as rollback_error:
                    self.logger.error(
                        f"Rollback failed for {symbol} order {order_id}: {rollback_error}",
                        exc_info=True,
                    )

            return {"success": False, "error": error_msg, "order_id": order_id}

    async def _handle_failed_closure(
        self, symbol: str, order_id: str, error: str
    ) -> None:
        """Handle failed position closure with emergency procedures."""
        try:
            # 1. Log the failure
            self.logger.error(
                f"Position closure failed for {symbol}",
                extra={"symbol": symbol, "order_id": order_id, "error": error},
            )

            # 2. Update position state to ERROR
            async with self.db_queries.transaction() as conn:
                await conn.execute(
                    """
                    UPDATE positions 
                    SET state = 'ERROR',
                        update_time = ?,
                        metadata = json_set(
                            COALESCE(metadata, '{}'),
                            '$.closure_error',
                            ?
                        )
                    WHERE symbol = ? AND state = 'CLOSING'
                    """,
                    [
                        int(time.time() * 1000),
                        json.dumps({"error": error, "order_id": order_id}),
                        symbol,
                    ],
                )

            # 3. Notify risk manager
            if hasattr(self.ctx, "risk_manager"):
                await self.ctx.risk_manager.handle_closure_failure(
                    symbol, order_id, error
                )

            # 4. Trigger circuit breaker
            if hasattr(self.ctx, "circuit_breaker"):
                await self.ctx.circuit_breaker.trigger_emergency_stop(
                    f"Position closure failed for {symbol}: {error}"
                )

            # 5. Schedule immediate reconciliation
            if hasattr(self.ctx, "reconciliation_manager"):
                await self.ctx.reconciliation_manager.schedule_immediate_reconciliation(
                    symbol, reason=f"Failed closure: {error}"
                )

        except Exception as e:
            self.logger.critical(
                f"Emergency handling failed for {symbol} closure: {e}", exc_info=True
            )

    async def create_order(
        self, symbol: str, order_type: str, side: str, amount: Decimal
    ) -> Optional[str]:
        """Create an order with proper resource management."""
        if not self.initialized and not await self.initialize():
            return None

        try:
            async with self.exchange_manager.rate_limiter.acquire_context("trade"):
                order = await self.exchange_manager.create_order(
                    symbol, side, amount, order_type=order_type
                )

                order_id = order.get("id")
                if not order_id:
                    raise ValidationError("Order response missing 'id'")

                # Log order if database is available
                if self.db_queries:
                    try:
                        await self.db_queries.store_trade(
                            {
                                "symbol": symbol,
                                "side": side,
                                "amount": str(amount),
                                "order_type": order_type,
                                "order_id": order_id,
                                "timestamp": datetime.utcnow().timestamp(),
                            }
                        )
                    except DatabaseError as e:
                        self.logger.error(f"Failed to log order: {e}")
                        # Continue since order was created successfully

                return order_id

        except (ExchangeError, ValidationError) as e:
            await handle_error_async(e, "ExchangeInterface.create_order", self.logger)
            return None
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.create_order", self.logger)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with proper resource management."""
        if not self.initialized and not await self.initialize():
            return False

        try:
            async with self.exchange_manager.rate_limiter.acquire_context("trade"):
                result = await self.exchange_manager.exchange.cancel_order(order_id)

                # Log cancellation if database is available
                if self.db_queries and result:
                    try:
                        await self.db_queries.update_trade(
                            order_id,
                            {
                                "status": "CANCELLED",
                                "cancel_time": datetime.utcnow().timestamp(),
                            },
                        )
                    except DatabaseError as e:
                        self.logger.error(f"Failed to log order cancellation: {e}")
                        # Continue since cancellation was successful

                return bool(result)

        except ExchangeError as e:
            await handle_error_async(e, "ExchangeInterface.cancel_order", self.logger)
            return False
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.cancel_order", self.logger)
            return False

    async def close(self) -> None:
        """Properly close all resources."""
        try:
            if self.exchange_manager:
                await self.exchange_manager.close()
            self.initialized = False
        except Exception as e:
            self.logger.error(f"Error during exchange interface cleanup: {e}")
            # Don't re-raise as this is cleanup code
