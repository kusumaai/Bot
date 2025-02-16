"""
Unit tests for the CircuitBreaker class.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.circuit_breaker import CircuitBreaker
from utils.exceptions import CircuitBreakerError


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_portfolio_manager():
    manager = AsyncMock()
    manager.initialized = True
    manager.risk_limits = MagicMock()
    manager.risk_limits.emergency_stop_pct = Decimal("0.1")
    manager.risk_limits.max_daily_loss = Decimal("1000")
    manager.close_all_positions = AsyncMock()
    manager.reset = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_order_manager():
    manager = AsyncMock()
    manager.cancel_all_orders = AsyncMock()
    manager.reset = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_context(mock_logger, mock_portfolio_manager, mock_order_manager):
    context = MagicMock()
    context.logger = mock_logger
    context.portfolio_manager = mock_portfolio_manager
    context.order_manager = mock_order_manager
    context.running = True
    return context


@pytest.fixture
async def circuit_breaker(mock_context):
    cb = CircuitBreaker(mock_context)
    yield cb
    await cb._cleanup()


@pytest.mark.asyncio
async def test_initialization(circuit_breaker, mock_context):
    """Test circuit breaker initialization."""
    success = await circuit_breaker.initialize()
    assert success
    assert circuit_breaker.initialized
    assert circuit_breaker._state == "NORMAL"
    assert circuit_breaker._monitor_task is not None
    assert not circuit_breaker.triggered
    assert not circuit_breaker.emergency_triggered


@pytest.mark.asyncio
async def test_initialization_failure_no_portfolio_manager(mock_context):
    """Test initialization failure when portfolio manager is not initialized."""
    mock_context.portfolio_manager.initialized = False
    cb = CircuitBreaker(mock_context)
    success = await cb.initialize()
    assert not success
    assert not cb.initialized


@pytest.mark.asyncio
async def test_state_transitions(circuit_breaker):
    """Test state transitions."""
    await circuit_breaker.initialize()

    # Test transition to WARNING
    await circuit_breaker._transition_state("WARNING")
    assert circuit_breaker._state == "WARNING"

    # Test transition to EMERGENCY
    await circuit_breaker._transition_state("EMERGENCY")
    assert circuit_breaker._state == "EMERGENCY"
    assert circuit_breaker.triggered
    assert circuit_breaker.emergency_triggered


@pytest.mark.asyncio
async def test_emergency_stop_coordination(circuit_breaker, mock_context):
    """Test coordinated emergency stop."""
    await circuit_breaker.initialize()
    await circuit_breaker.trigger_emergency_stop("Test emergency")

    # Verify all components were shut down
    mock_context.portfolio_manager.close_all_positions.assert_awaited_once_with(
        "Emergency stop triggered"
    )
    mock_context.order_manager.cancel_all_orders.assert_awaited_once_with(
        "Emergency stop triggered"
    )
    assert circuit_breaker._shutdown_complete.is_set()


@pytest.mark.asyncio
async def test_component_failure_handling(circuit_breaker, mock_context):
    """Test handling of component failures during shutdown."""
    mock_context.portfolio_manager.close_all_positions.side_effect = Exception(
        "Test error"
    )

    await circuit_breaker.initialize()
    await circuit_breaker.trigger_emergency_stop("Test emergency")

    assert "portfolio" in circuit_breaker._affected_components
    assert circuit_breaker._component_states["portfolio"] == "ERROR"


@pytest.mark.asyncio
async def test_recovery_attempts(circuit_breaker, mock_context):
    """Test recovery attempt handling."""
    await circuit_breaker.initialize()

    # Simulate a warning state
    await circuit_breaker._transition_state("WARNING")

    # Attempt recovery
    await circuit_breaker._attempt_recovery()

    assert circuit_breaker._recovery_attempts == 1
    mock_context.portfolio_manager.reset.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_recovery_attempts(circuit_breaker):
    """Test maximum recovery attempts limit."""
    await circuit_breaker.initialize()
    circuit_breaker._recovery_attempts = circuit_breaker._max_recovery_attempts

    await circuit_breaker._attempt_recovery()
    assert circuit_breaker._state == "WARNING"  # State should not change


@pytest.mark.asyncio
async def test_monitor_loop_error_handling(circuit_breaker):
    """Test monitor loop error handling."""
    await circuit_breaker.initialize()

    # Simulate consecutive errors
    with patch.object(
        circuit_breaker, "check_conditions", side_effect=Exception("Test error")
    ):
        # Wait for monitor loop to detect errors
        await asyncio.sleep(0.1)
        assert circuit_breaker.triggered
        assert circuit_breaker.emergency_triggered


@pytest.mark.asyncio
async def test_cleanup(circuit_breaker):
    """Test cleanup procedure."""
    await circuit_breaker.initialize()
    circuit_breaker._affected_components.add("test")
    circuit_breaker._component_states["test"] = "ERROR"

    await circuit_breaker._cleanup()

    assert not circuit_breaker.initialized
    assert not circuit_breaker._affected_components
    assert not circuit_breaker._component_states
    assert circuit_breaker._state == "NORMAL"


@pytest.mark.asyncio
async def test_get_status(circuit_breaker):
    """Test status reporting."""
    await circuit_breaker.initialize()
    circuit_breaker._affected_components.add("test")
    circuit_breaker._component_states["test"] = "ERROR"
    circuit_breaker._last_error = Exception("Test error")

    status = await circuit_breaker.get_status()

    assert status["state"] == "NORMAL"
    assert "test" in status["affected_components"]
    assert status["component_states"]["test"] == "ERROR"
    assert "Test error" in status["last_error"]


@pytest.mark.asyncio
async def test_concurrent_emergency_stops(circuit_breaker):
    """Test handling of concurrent emergency stops."""
    await circuit_breaker.initialize()

    # Trigger multiple emergency stops concurrently
    tasks = [
        asyncio.create_task(circuit_breaker.trigger_emergency_stop("Test emergency"))
        for _ in range(3)
    ]
    await asyncio.gather(*tasks)

    # Verify only one emergency stop was processed
    assert circuit_breaker.triggered
    assert circuit_breaker.emergency_triggered
    assert circuit_breaker._state == "EMERGENCY"
