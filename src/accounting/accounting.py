import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict

from bot_types.base_types import ValidationResult
from trading.position import Position  # Use single source Position
from utils.error_handler import ValidationError, handle_error_async


async def validate_account(signal: Dict[str, Any], ctx: Any) -> ValidationResult:
    """
    Validate account health and balance requirements.

    Args:
        signal: Trading signal containing exchange info
        ctx: Trading context with configuration

    Returns:
        ValidationResult with validation status and any error messages
    """
    try:
        # Validate input signal
        if not isinstance(signal, dict) or "exchange" not in signal:
            return ValidationResult(False, "Invalid signal format")
        if not signal.get("exchange"):
            return ValidationResult(False, "Missing exchange identifier")

        # Get configuration thresholds with safe defaults
        min_free_balance = Decimal(
            str(ctx.config.get("min_free_balance_threshold", "100.0"))
        )
        min_health_ratio = Decimal(
            str(ctx.config.get("min_account_health_ratio", "2.0"))
        )
        absolute_minimum = Decimal(
            str(ctx.config.get("absolute_minimum_balance", "50.0"))
        )
        max_used_ratio = Decimal(str(ctx.config.get("max_used_balance_ratio", "0.8")))
        max_drawdown = Decimal(
            str(ctx.config.get("max_drawdown", "0.2"))
        )  # 20% max drawdown
        max_daily_loss = Decimal(
            str(ctx.config.get("max_daily_loss", "0.1"))
        )  # 10% max daily loss
        min_margin_level = Decimal(
            str(ctx.config.get("min_margin_level", "1.5"))
        )  # 150% minimum margin level
        max_position_count = int(ctx.config.get("max_position_count", 5))
        emergency_stop_pct = Decimal(
            str(ctx.config.get("emergency_stop_pct", "0.15"))
        )  # 15% emergency stop

        # Validate configuration values
        if min_free_balance <= 0:
            return ValidationResult(
                False, "min_free_balance_threshold must be positive"
            )
        if min_health_ratio <= Decimal("1.0"):
            return ValidationResult(
                False, "min_account_health_ratio must be greater than 1.0"
            )
        if absolute_minimum <= 0:
            return ValidationResult(False, "absolute_minimum_balance must be positive")
        if max_used_ratio <= 0 or max_used_ratio >= Decimal("1.0"):
            return ValidationResult(
                False, "max_used_balance_ratio must be between 0 and 1"
            )
        if max_drawdown <= 0 or max_drawdown >= Decimal("1.0"):
            return ValidationResult(False, "max_drawdown must be between 0 and 1")
        if max_daily_loss <= 0 or max_daily_loss >= Decimal("1.0"):
            return ValidationResult(False, "max_daily_loss must be between 0 and 1")
        if emergency_stop_pct >= max_drawdown:
            return ValidationResult(
                False, "emergency_stop_pct must be less than max_drawdown"
            )

        # Validate trading mode and requirements
        if not ctx.config.get("paper_mode", False):
            # Stricter requirements for live trading
            absolute_minimum = max(
                absolute_minimum, Decimal("100.0")
            )  # Higher minimum for live
            min_margin_level = max(
                min_margin_level, Decimal("2.0")
            )  # 200% minimum margin for live
            max_position_count = min(
                max_position_count, 3
            )  # More conservative position limit

        async with ctx.db_connection.transaction() as conn:
            # Query with explicit column selection and NULL handling
            query = """
                SELECT 
                    COALESCE(balance, 0) as balance,
                    COALESCE(used_balance, 0) as used_balance,
                    COALESCE(locked_balance, 0) as locked_balance,
                    COALESCE(margin_level, 0) as margin_level,
                    COALESCE(unrealized_pnl, 0) as unrealized_pnl,
                    COALESCE(daily_realized_pnl, 0) as daily_realized_pnl,
                    COALESCE(peak_balance, 0) as peak_balance,
                    COALESCE(position_count, 0) as position_count,
                    last_update_time,
                    last_sync_time
                FROM account 
                WHERE exchange = ?
                """
            row = await conn.execute_one(query, [signal["exchange"]])

            if not row:
                return ValidationResult(
                    False, f"No account found for exchange {signal['exchange']}"
                )

            # Convert to Decimal with validation
            try:
                balance = Decimal(str(row["balance"]))
                used_balance = Decimal(str(row["used_balance"]))
                locked_balance = Decimal(str(row["locked_balance"]))
                margin_level = Decimal(str(row["margin_level"]))
                unrealized_pnl = Decimal(str(row["unrealized_pnl"]))
                daily_realized_pnl = Decimal(str(row["daily_realized_pnl"]))
                peak_balance = Decimal(str(row["peak_balance"]))
                position_count = int(row["position_count"])
                last_update = row["last_update_time"]
                last_sync = row["last_sync_time"]
            except (TypeError, InvalidOperation) as e:
                return ValidationResult(
                    False, f"Invalid balance values in database: {e}"
                )

            # Validate positive balances
            if balance < 0:
                return ValidationResult(False, f"Negative total balance: {balance}")
            if used_balance < 0:
                return ValidationResult(False, f"Negative used balance: {used_balance}")
            if locked_balance < 0:
                return ValidationResult(
                    False, f"Negative locked balance: {locked_balance}"
                )

            # Calculate key health metrics
            free_balance = balance - used_balance - locked_balance
            used_ratio = (
                (used_balance + locked_balance) / balance
                if balance > 0
                else Decimal("1.0")
            )
            health_ratio = (
                balance / (used_balance + locked_balance)
                if (used_balance + locked_balance) > 0
                else Decimal("999.0")
            )

            # Calculate drawdown metrics
            current_drawdown = (
                (peak_balance - balance) / peak_balance
                if peak_balance > 0
                else Decimal("0")
            )
            total_pnl = unrealized_pnl + daily_realized_pnl
            daily_loss_pct = (
                abs(daily_realized_pnl) / balance
                if balance > 0 and daily_realized_pnl < 0
                else Decimal("0")
            )

            # Validate data freshness
            max_staleness = ctx.config.get("max_balance_staleness", 300)
            max_sync_delay = ctx.config.get("max_sync_delay", 60)
            current_time = time.time()

            if current_time - last_update > max_staleness:
                return ValidationResult(
                    False, f"Account balance data is stale (last update: {last_update})"
                )
            if current_time - last_sync > max_sync_delay:
                return ValidationResult(
                    False, f"Account sync is delayed (last sync: {last_sync})"
                )

            # Comprehensive health checks
            if balance < absolute_minimum:
                return ValidationResult(
                    False,
                    f"Total balance {balance} below absolute minimum {absolute_minimum}",
                )

            if free_balance < min_free_balance:
                return ValidationResult(
                    False,
                    f"Free balance {free_balance} below minimum {min_free_balance}",
                )

            if used_ratio > max_used_ratio:
                return ValidationResult(
                    False,
                    f"Used balance ratio {used_ratio:.2%} exceeds maximum {max_used_ratio:.2%}",
                )

            if health_ratio < min_health_ratio:
                return ValidationResult(
                    False,
                    f"Account health ratio {health_ratio:.2f} below minimum {min_health_ratio:.2f}",
                )

            # Position and margin checks
            if position_count >= max_position_count:
                return ValidationResult(
                    False,
                    f"Position count {position_count} exceeds maximum {max_position_count}",
                )

            if margin_level < min_margin_level:
                return ValidationResult(
                    False,
                    f"Margin level {margin_level:.2%} below minimum {min_margin_level:.2%}",
                )

            # Drawdown and loss checks
            if current_drawdown >= max_drawdown:
                return ValidationResult(
                    False,
                    f"Current drawdown {current_drawdown:.2%} exceeds maximum {max_drawdown:.2%}",
                )

            if daily_loss_pct >= max_daily_loss:
                return ValidationResult(
                    False,
                    f"Daily loss {daily_loss_pct:.2%} exceeds maximum {max_daily_loss:.2%}",
                )

            if current_drawdown >= emergency_stop_pct:
                return ValidationResult(
                    False,
                    f"Emergency stop triggered: drawdown {current_drawdown:.2%} exceeds limit {emergency_stop_pct:.2%}",
                )

            # Validate balance relationships
            if used_balance + locked_balance > balance:
                return ValidationResult(
                    False, "Used + locked balance exceeds total balance"
                )

            # Validate unrealized PnL sanity
            max_unrealized_pnl_ratio = Decimal("0.5")  # 50% of balance
            if abs(unrealized_pnl) > balance * max_unrealized_pnl_ratio:
                return ValidationResult(
                    False,
                    f"Unrealized PnL {unrealized_pnl} exceeds {max_unrealized_pnl_ratio:.0%} of balance",
                )

            # Log account health metrics
            ctx.logger.info(
                "Account health check passed",
                extra={
                    "free_balance": str(free_balance),
                    "health_ratio": str(health_ratio),
                    "used_ratio": str(used_ratio),
                    "margin_level": str(margin_level),
                    "drawdown": str(current_drawdown),
                    "daily_loss": str(daily_loss_pct),
                    "position_count": position_count,
                    "exchange": signal["exchange"],
                },
            )

            return ValidationResult(True)

    except Exception as e:
        await handle_error_async(e, "validate_account")
        return ValidationResult(False, str(e))
