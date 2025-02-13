import pytest
from decimal import Decimal
from unittest.mock import MagicMock
import logging
import asyncio
import time

from trading.portfolio import PortfolioManager
from trading.position import Position
from utils.error_handler import PortfolioError
from risk.limits import RiskLimits


@pytest.fixture
def risk_limits():
    """Provide test risk limits."""
    return RiskLimits.from_config({
        'max_positions': 5,
        'max_position_size': '0.5',
        'max_leverage': '3.0'
    })


@pytest.fixture
def mock_logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def portfolio_manager(risk_limits, mock_logger):
    """Provide a PortfolioManager instance with mocked dependencies."""
    ctx = MagicMock()
    ctx.logger = mock_logger
    ctx.config = {
        "portfolio": {
            "initial_balance": "10000"
        }
    }
    ctx.risk_limits = risk_limits
    portfolio_mgr = PortfolioManager(ctx)
    asyncio.run(portfolio_mgr.initialize())
    return portfolio_mgr


@pytest.mark.asyncio
async def test_portfolio_initialization(portfolio_manager):
    """Test initializing the PortfolioManager."""
    assert portfolio_manager.initialized is True
    assert portfolio_manager.current_balance == Decimal('10000')


@pytest.mark.asyncio
async def test_add_position_within_limits(portfolio_manager):
    """Test adding a position within risk limits."""
    position = Position(
        symbol='BTC/USDT',
        side='buy',
        entry_price=Decimal('50000'),
        size=Decimal('0.1'),
        timestamp=int(time.time())
    )
    result = await portfolio_manager.add_position(position)
    assert result is True
    assert len(portfolio_manager.positions) == 1


@pytest.mark.asyncio
async def test_add_position_exceeds_max_positions(portfolio_manager):
    """Test adding a position that exceeds the maximum number of positions."""
    for i in range(5):
        position = Position(
            symbol=f'BTC/USDT-{i}',
            side='buy',
            entry_price=Decimal('50000'),
            size=Decimal('0.1'),
            timestamp=int(time.time())
        )
        result = await portfolio_manager.add_position(position)
        assert result is True
    
    # Attempt to add a sixth position
    position = Position(
        symbol='BTC/USDT-5',
        side='buy',
        entry_price=Decimal('50000'),
        size=Decimal('0.1'),
        timestamp=int(time.time())
    )
    result = await portfolio_manager.add_position(position)
    assert result is False
    assert portfolio_manager.logger.error.called


def test_add_position_exceeds_max_size(portfolio_manager):
    """Test adding a position exceeding the maximum allowed size."""
    large_position = Position(
        id='pos007',
        symbol='BTC/USDT',
        direction='long',
        size=Decimal('1.0'),  # Exceeds max_position_size
        entry_price=Decimal('50000')
    )
    result = portfolio_manager.add_position(large_position)
    assert result is False
    portfolio_manager.logger.warning.assert_called_with(
        "Cannot add position pos007: Position size 1.0 exceeds maximum allowed 0.5."
    )


def test_remove_position_success(portfolio_manager):
    """Test successful removal of a position."""
    position = Position(
        id='pos008',
        symbol='ETH/USDT',
        direction='short',
        size=Decimal('0.2'),
        entry_price=Decimal('3000')
    )
    portfolio_manager.add_position(position)
    assert len(portfolio_manager.positions) == 1

    result = portfolio_manager.remove_position('pos008')
    assert result is True
    assert len(portfolio_manager.positions) == 0


def test_remove_position_not_found(portfolio_manager):
    """Test removal of a non-existent position."""
    result = portfolio_manager.remove_position('nonexistent_pos')
    assert result is False
    portfolio_manager.logger.warning.assert_called_with(
        "Cannot remove position nonexistent_pos: Position not found."
    )


def test_get_total_exposure(portfolio_manager):
    """Test calculation of total portfolio exposure."""
    position1 = Position(
        id='pos009',
        symbol='BTC/USDT',
        direction='long',
        size=Decimal('0.1'),
        entry_price=Decimal('50000')
    )
    position2 = Position(
        id='pos010',
        symbol='ETH/USDT',
        direction='short',
        size=Decimal('0.2'),
        entry_price=Decimal('3000')
    )
    portfolio_manager.add_position(position1)
    portfolio_manager.add_position(position2)

    total_exposure = portfolio_manager.get_total_exposure()
    expected_exposure = (Decimal('0.1') * Decimal('50000')) + (Decimal('0.2') * Decimal('3000'))  # 5000 + 600 = 5600
    assert total_exposure == Decimal('5600') 