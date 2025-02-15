#! /usr/bin/env python3
"""
Module: tests.unit.test_execution_flow
Comprehensive testing of trading execution flow including order management,
execution cycles, and main trading loop functionality.
"""
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.error_handler import ExecutionError, OrderError


class TestExecutionFlow:
    """Test suite for trading execution flow."""

    @pytest.fixture
    async def execution_manager(self, trading_context):
        """Provide configured execution manager."""
        from execution.manager import ExecutionManager

        manager = ExecutionManager(trading_context)
        await manager.initialize()
        return manager

    @pytest.fixture
    def mock_order(self):
        """Provide a mock order for testing."""
        return {
            "id": "test_order_1",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": Decimal("0.1"),
            "price": Decimal("50000"),
            "type": "limit",
            "status": "open",
        }

    @pytest.mark.asyncio
    async def test_order_execution_cycle(self, execution_manager, mock_order):
        """Test complete order execution cycle."""
        # Prepare order
        execution_manager.validate_order = AsyncMock(return_value=True)
        execution_manager.risk_check = AsyncMock(return_value=True)
        execution_manager.place_order = AsyncMock(return_value=mock_order)

        # Execute order
        result = await execution_manager.execute_order(mock_order)
        assert result["status"] == "open"
        assert result["id"] == mock_order["id"]

        # Update order status
        updated_order = mock_order.copy()
        updated_order["status"] = "filled"
        execution_manager.get_order_status = AsyncMock(return_value=updated_order)

        status = await execution_manager.check_order_status(mock_order["id"])
        assert status == "filled"

    @pytest.mark.asyncio
    async def test_execution_validation(self, execution_manager, mock_order):
        """Test execution validation checks."""
        # Invalid order type
        invalid_order = mock_order.copy()
        invalid_order["type"] = "invalid"
        with pytest.raises(OrderError):
            await execution_manager.validate_order(invalid_order)

        # Invalid amount
        invalid_order = mock_order.copy()
        invalid_order["amount"] = Decimal("0")
        with pytest.raises(OrderError):
            await execution_manager.validate_order(invalid_order)

        # Invalid price
        invalid_order = mock_order.copy()
        invalid_order["price"] = Decimal("-1")
        with pytest.raises(OrderError):
            await execution_manager.validate_order(invalid_order)

    @pytest.mark.asyncio
    async def test_risk_checks(self, execution_manager, mock_order):
        """Test risk management in execution flow."""
        # Setup risk manager mock
        execution_manager.risk_manager.validate_order = AsyncMock(return_value=True)
        assert await execution_manager.risk_check(mock_order)

        # Test risk limit violation
        execution_manager.risk_manager.validate_order = AsyncMock(return_value=False)
        with pytest.raises(ExecutionError):
            await execution_manager.risk_check(mock_order)

    @pytest.mark.asyncio
    async def test_execution_retry_logic(self, execution_manager, mock_order):
        """Test execution retry mechanism."""
        # Setup failing order placement that succeeds on retry
        attempt_count = 0

        async def mock_place_order(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ExecutionError("Temporary failure")
            return mock_order

        execution_manager.place_order = mock_place_order
        result = await execution_manager.execute_with_retry(mock_order)

        assert result == mock_order
        assert attempt_count == 3  # Verify it took 3 attempts

    @pytest.mark.asyncio
    async def test_main_execution_loop(self, execution_manager):
        """Test main execution loop functionality."""
        # Setup execution conditions
        execution_count = 0

        async def mock_execution_cycle():
            nonlocal execution_count
            execution_count += 1
            if execution_count >= 3:
                execution_manager.stop()

        # Run execution loop
        execution_manager.execute_cycle = mock_execution_cycle
        await execution_manager.start()

        assert execution_count == 3  # Verify execution cycles completed

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, execution_manager, mock_order):
        """Test handling of concurrent order execution."""
        # Setup concurrent orders
        orders = [
            mock_order,
            {**mock_order, "id": "test_order_2"},
            {**mock_order, "id": "test_order_3"},
        ]

        # Execute orders concurrently
        results = await asyncio.gather(
            *[execution_manager.execute_order(order) for order in orders]
        )

        assert len(results) == 3
        assert all(r["status"] in ["open", "filled"] for r in results)

    @pytest.mark.asyncio
    async def test_execution_error_handling(self, execution_manager, mock_order):
        """Test error handling during execution."""
        # Test network error
        execution_manager.place_order = AsyncMock(
            side_effect=Exception("Network error")
        )
        with pytest.raises(ExecutionError) as exc_info:
            await execution_manager.execute_order(mock_order)
        assert "Network error" in str(exc_info.value)

        # Test exchange error
        execution_manager.place_order = AsyncMock(
            side_effect=Exception("Exchange rejected order")
        )
        with pytest.raises(ExecutionError) as exc_info:
            await execution_manager.execute_order(mock_order)
        assert "Exchange rejected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execution_state_management(self, execution_manager):
        """Test execution state management."""
        # Test state transitions
        assert not execution_manager.is_running()

        await execution_manager.start()
        assert execution_manager.is_running()

        await execution_manager.pause()
        assert not execution_manager.is_running()
        assert execution_manager.is_paused()

        await execution_manager.resume()
        assert execution_manager.is_running()
        assert not execution_manager.is_paused()

        await execution_manager.stop()
        assert not execution_manager.is_running()
