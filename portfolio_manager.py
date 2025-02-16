from decimal import Decimal
from typing import Optional

from trading.position import Position
from utils.exceptions import ConcurrentModificationError


class PortfolioManager:
    def __init__(self, logger, db):
        self.logger = logger
        self.db = db
        self.positions = {}
        self._lock = None  # Assuming a lock is set up

    async def close_position(
        self, symbol: str, exit_price: Decimal
    ) -> Optional[Position]:
        # First get position data under lock
        position = None
        position_version = None

        async with self._lock:
            if symbol not in self.positions:
                self.logger.warning(f"No existing position for {symbol}")
                return None

            position = self.positions[symbol]
            if position.closed:
                self.logger.warning(f"Position already closed: {symbol}")
                return None

            # Store version for optimistic locking
            position_version = position.version

        if not position:
            return None

        try:
            # Perform slow operations outside lock
            updated_position = position.copy()  # Create copy for updates
            await updated_position.update(exit_price)

            if self.db:
                async with self.db.transaction():
                    # Verify position hasn't changed
                    current_position = await self.db.get_position(symbol)
                    if current_position.version != position_version:
                        raise ConcurrentModificationError(
                            f"Position {symbol} was modified concurrently"
                        )

                    # Perform database updates
                    await updated_position.close_position(
                        updated_position.unrealized_pnl
                    )
                    await self._update_position_db(updated_position, closed=True)

                    # Final state update under lock
                    async with self._lock:
                        if symbol in self.positions:
                            # Verify version again
                            if self.positions[symbol].version != position_version:
                                raise ConcurrentModificationError(
                                    f"Position {symbol} was modified concurrently"
                                )
                            self.positions[symbol] = updated_position
                            del self.positions[symbol]

                self.logger.info(
                    f"Successfully closed position for {symbol} at {exit_price}"
                )
                return updated_position

        except ConcurrentModificationError as e:
            self.logger.warning(f"Concurrent modification detected: {e}")
            # Retry logic could be added here
            return None
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {e}")
            # Ensure we clean up any partial state
            async with self._lock:
                if symbol in self.positions:
                    self.positions[symbol].status = Position.ERROR
            return None

    # Helper method for database updates
    async def _update_position_db(
        self, position: Position, closed: bool = False
    ) -> None:
        """Update position in database without holding portfolio lock"""
        if not self.db:
            return

        try:
            await self.db.execute(
                """
                UPDATE positions 
                SET exit_price = ?, 
                    closed = ?,
                    realized_pnl = ?,
                    status = ?,
                    version = version + 1
                WHERE symbol = ? AND version = ?
                """,
                (
                    str(position.exit_price),
                    closed,
                    str(position.realized_pnl),
                    position.status.value,
                    position.symbol,
                    position.version,
                ),
            )
        except Exception as e:
            self.logger.error(
                f"Database update failed for position {position.symbol}: {e}"
            )
            raise
