#! /usr/bin/env python3
"""
Module: tests.unit.test_risk_management
Comprehensive testing of risk management functionality including position sizing,
risk limits, and edge cases.
"""
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from risk.limits import RiskLimits
from risk.manager import RiskManager
from utils.error_handler import RiskError


class TestRiskManagement:
    """Test suite for risk management functionality."""

    @pytest.fixture
    def risk_limits(self):
        """Provide standard risk limits for testing."""
        return RiskLimits(
            max_position_size=Decimal("1.0"),
            min_position_size=Decimal("0.01"),
            max_leverage=Decimal("3"),
            max_drawdown=Decimal("20"),
            max_daily_loss=Decimal("5"),
            max_positions=5,
            position_risk_limit=Decimal("2"),
            correlation_limit=Decimal("0.7"),
        )

    @pytest.fixture
    async def risk_manager(self, trading_context, risk_limits):
        """Provide configured risk manager for testing."""
        manager = RiskManager(trading_context)
        manager.limits = risk_limits
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_position_size_calculation(self, risk_manager):
        """Test position size calculation with various inputs."""
        # Test normal case
        size = await risk_manager.calculate_position_size(
            price=Decimal("50000"),
            account_size=Decimal("100000"),
            risk_per_trade=Decimal("1"),
        )
        assert Decimal("0.01") <= size <= Decimal("1.0")

        # Test minimum position size enforcement
        small_size = await risk_manager.calculate_position_size(
            price=Decimal("50000"),
            account_size=Decimal("1000"),
            risk_per_trade=Decimal("0.1"),
        )
        assert small_size >= Decimal("0.01")

        # Test maximum position size enforcement
        large_size = await risk_manager.calculate_position_size(
            price=Decimal("50000"),
            account_size=Decimal("1000000"),
            risk_per_trade=Decimal("5"),
        )
        assert large_size <= Decimal("1.0")

    @pytest.mark.asyncio
    async def test_risk_metrics_validation(self, risk_manager):
        """Test validation of various risk metrics."""
        # Valid metrics
        valid_metrics = {
            "drawdown": Decimal("15"),
            "daily_loss": Decimal("3"),
            "position_count": 3,
            "leverage": Decimal("2"),
        }
        assert await risk_manager.validate_risk_metrics(valid_metrics)

        # Test drawdown limit
        invalid_metrics = valid_metrics.copy()
        invalid_metrics["drawdown"] = Decimal("25")
        with pytest.raises(RiskError) as exc_info:
            await risk_manager.validate_risk_metrics(invalid_metrics)
        assert "Drawdown exceeds maximum" in str(exc_info.value)

        # Test daily loss limit
        invalid_metrics["drawdown"] = Decimal("15")
        invalid_metrics["daily_loss"] = Decimal("6")
        with pytest.raises(RiskError) as exc_info:
            await risk_manager.validate_risk_metrics(invalid_metrics)
        assert "Daily loss exceeds maximum" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_position_correlation(self, risk_manager):
        """Test position correlation checks."""
        # Setup test positions
        positions = [
            {"symbol": "BTC/USDT", "size": Decimal("0.1")},
            {"symbol": "ETH/USDT", "size": Decimal("1.0")},
        ]

        # Test correlation calculation
        correlation = await risk_manager.calculate_position_correlation(
            positions[0]["symbol"], positions[1]["symbol"]
        )
        assert correlation <= Decimal("1.0")

        # Test correlation limit enforcement
        risk_manager.get_correlation = AsyncMock(return_value=Decimal("0.8"))
        with pytest.raises(RiskError) as exc_info:
            await risk_manager.validate_new_position("ETH/USDT", Decimal("0.1"))
        assert "Correlation too high" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_portfolio_risk(self, risk_manager):
        """Test portfolio-wide risk management."""
        # Test position count limit
        positions = [
            {"symbol": f"COIN{i}/USDT", "size": Decimal("0.1")} for i in range(6)
        ]
        with pytest.raises(RiskError) as exc_info:
            await risk_manager.validate_portfolio_risk(positions)
        assert "Maximum positions exceeded" in str(exc_info.value)

        # Test portfolio value limit
        large_positions = [
            {"symbol": "BTC/USDT", "size": Decimal("2.0")},
            {"symbol": "ETH/USDT", "size": Decimal("20.0")},
        ]
        with pytest.raises(RiskError) as exc_info:
            await risk_manager.validate_portfolio_risk(large_positions)
        assert "Portfolio value exceeds limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_emergency_risk_handling(self, risk_manager):
        """Test emergency risk handling scenarios."""
        # Simulate market crash
        crash_metrics = {
            "drawdown": Decimal("30"),
            "daily_loss": Decimal("10"),
            "position_count": 3,
            "leverage": Decimal("1"),
        }

        # Test emergency protocols
        with pytest.raises(RiskError) as exc_info:
            await risk_manager.validate_risk_metrics(crash_metrics)
        assert "Emergency stop triggered" in str(exc_info.value)

        # Verify all trading is blocked
        with pytest.raises(RiskError):
            await risk_manager.validate_new_position("BTC/USDT", Decimal("0.1"))

    @pytest.mark.asyncio
    async def test_risk_limit_updates(self, risk_manager):
        """Test dynamic risk limit updates."""
        # Test limit adjustment based on performance
        await risk_manager.update_risk_limits(performance_factor=Decimal("1.2"))
        assert risk_manager.limits.position_risk_limit > Decimal("2")

        # Test limit reduction after losses
        await risk_manager.update_risk_limits(performance_factor=Decimal("0.8"))
        assert risk_manager.limits.position_risk_limit < Decimal("2")

        # Test limits don't exceed absolute maximums
        await risk_manager.update_risk_limits(performance_factor=Decimal("2.0"))
        assert risk_manager.limits.max_leverage <= Decimal("3")
