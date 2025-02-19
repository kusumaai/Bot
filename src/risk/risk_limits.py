import logging
from decimal import Decimal
from typing import Dict, Optional


class RiskLimits:
    """Class to manage trading risk limits."""

    def __init__(
        self,
        min_position_size: Decimal,
        max_position_size: Decimal,
        max_leverage: Decimal,
        max_drawdown: Decimal,
        max_daily_trades: int,
        max_open_positions: int,
        max_position_value: Decimal,
        emergency_stop_pct: Decimal,
        risk_factor: Decimal = Decimal("0.02"),
        max_risk_per_trade: Decimal = Decimal("0.01"),
        max_correlation: Decimal = Decimal("0.7"),
        min_liquidity: Decimal = Decimal("1000"),
        max_volatility: Decimal = Decimal("0.5"),
    ):
        """Initialize risk limits."""
        self.logger = logging.getLogger(__name__)

        # Position limits
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.max_daily_trades = max_daily_trades
        self.max_open_positions = max_open_positions
        self.max_position_value = max_position_value
        self.emergency_stop_pct = emergency_stop_pct

        # Risk parameters
        self.risk_factor = risk_factor
        self.max_risk_per_trade = max_risk_per_trade

        # Market condition limits
        self.max_correlation = max_correlation
        self.min_liquidity = min_liquidity
        self.max_volatility = max_volatility

    @classmethod
    def from_dict(cls, config: Dict) -> "RiskLimits":
        """Create RiskLimits from a dictionary configuration."""
        try:
            return cls(
                min_position_size=Decimal(
                    str(config.get("min_position_size", "0.001"))
                ),
                max_position_size=Decimal(str(config.get("max_position_size", "1.0"))),
                max_leverage=Decimal(str(config.get("max_leverage", "3.0"))),
                max_drawdown=Decimal(str(config.get("max_drawdown", "0.1"))),
                max_daily_trades=int(config.get("max_daily_trades", 10)),
                max_open_positions=int(config.get("max_open_positions", 5)),
                max_position_value=Decimal(
                    str(config.get("max_position_value", "1000.0"))
                ),
                emergency_stop_pct=Decimal(
                    str(config.get("emergency_stop_pct", "0.15"))
                ),
                risk_factor=Decimal(str(config.get("risk_factor", "0.02"))),
                max_risk_per_trade=Decimal(
                    str(config.get("max_risk_per_trade", "0.01"))
                ),
                max_correlation=Decimal(str(config.get("max_correlation", "0.7"))),
                min_liquidity=Decimal(str(config.get("min_liquidity", "1000"))),
                max_volatility=Decimal(str(config.get("max_volatility", "0.5"))),
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to create RiskLimits from config: {e}"
            )
            raise ValueError(f"Invalid risk limits configuration: {e}")

    def update(self, updates: Dict) -> None:
        """Update risk limits from a dictionary of updates."""
        try:
            for key, value in updates.items():
                if hasattr(self, key):
                    if isinstance(getattr(self, key), Decimal):
                        setattr(self, key, Decimal(str(value)))
                    elif isinstance(getattr(self, key), int):
                        setattr(self, key, int(value))
                    else:
                        setattr(self, key, value)
                    self.logger.info(f"Updated {key} to {value}")
                else:
                    self.logger.warning(f"Unknown risk limit parameter: {key}")
        except Exception as e:
            self.logger.error(f"Failed to update risk limits: {e}")
            raise ValueError(f"Invalid risk limit update: {e}")

    def to_dict(self) -> Dict:
        """Convert risk limits to a dictionary."""
        return {
            "min_position_size": str(self.min_position_size),
            "max_position_size": str(self.max_position_size),
            "max_leverage": str(self.max_leverage),
            "max_drawdown": str(self.max_drawdown),
            "max_daily_trades": self.max_daily_trades,
            "max_open_positions": self.max_open_positions,
            "max_position_value": str(self.max_position_value),
            "emergency_stop_pct": str(self.emergency_stop_pct),
            "risk_factor": str(self.risk_factor),
            "max_risk_per_trade": str(self.max_risk_per_trade),
            "max_correlation": str(self.max_correlation),
            "min_liquidity": str(self.min_liquidity),
            "max_volatility": str(self.max_volatility),
        }
