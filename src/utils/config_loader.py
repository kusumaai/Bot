#! /usr/bin/env python3
# src/utils/config_loader.py
"""
Module: src.utils
Provides configuration loading functionality.
"""
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

from bot_types.base_types import ValidationResult
from utils.numeric_handler import NumericHandler


# risk config class that represents the risk config
@dataclass
class RiskConfig:
    max_position_size: Decimal
    emergency_stop_pct: Decimal
    max_drawdown: Decimal
    max_daily_loss: Decimal
    max_positions: int
    min_trade_size: Decimal = Decimal("0.0001")  # Minimum trade size
    max_leverage: Decimal = Decimal("3.0")  # Maximum allowed leverage
    max_slippage_pct: Decimal = Decimal("0.01")  # Maximum allowed slippage
    position_timeout: int = 7200  # Maximum position hold time in seconds
    min_profit_pct: Decimal = Decimal("0.001")  # Minimum profit target
    max_correlation: Decimal = Decimal("0.7")  # Maximum portfolio correlation
    max_open_trades: int = 5  # Maximum concurrent open trades

    # from dict method
    @classmethod
    def from_dict(cls, data: Dict) -> "RiskConfig":
        nh = NumericHandler()
        return cls(
            max_position_size=nh.percentage_to_decimal(data["max_position_size"]),
            emergency_stop_pct=nh.percentage_to_decimal(data["emergency_stop_pct"]),
            max_drawdown=nh.percentage_to_decimal(data["max_drawdown"]),
            max_daily_loss=nh.percentage_to_decimal(data["max_daily_loss"]),
            max_positions=int(data["max_positions"]),
            min_trade_size=nh.to_decimal(data.get("min_trade_size", "0.0001")),
            max_leverage=nh.to_decimal(data.get("max_leverage", "3.0")),
            max_slippage_pct=nh.percentage_to_decimal(
                data.get("max_slippage_pct", "0.01")
            ),
            position_timeout=int(data.get("position_timeout", 7200)),
            min_profit_pct=nh.percentage_to_decimal(
                data.get("min_profit_pct", "0.001")
            ),
            max_correlation=nh.to_decimal(data.get("max_correlation", "0.7")),
            max_open_trades=int(data.get("max_open_trades", 5)),
        )

    def validate(self) -> "ValidationResult":
        """Validate the risk config with comprehensive checks"""
        result = ValidationResult(is_valid=True)

        # Basic range validations
        if self.max_position_size > Decimal("0.5"):
            result.add_error("max_position_size cannot exceed 50%")
        if self.max_drawdown > Decimal("0.2"):
            result.add_error("max_drawdown cannot exceed 20%")

        # Validate emergency stop
        if self.emergency_stop_pct >= self.max_drawdown:
            result.add_error("emergency_stop_pct must be less than max_drawdown")
        if self.emergency_stop_pct <= Decimal("0"):
            result.add_error("emergency_stop_pct must be positive")

        # Validate position limits
        if self.max_positions < 1:
            result.add_error("max_positions must be at least 1")
        if self.max_positions > 100:  # Reasonable upper limit
            result.add_error("max_positions cannot exceed 100")

        # Validate trade size limits
        if self.min_trade_size <= 0:
            result.add_error("min_trade_size must be positive")
        if self.min_trade_size >= self.max_position_size:
            result.add_error("min_trade_size must be less than max_position_size")

        # Validate leverage
        if self.max_leverage <= 0:
            result.add_error("max_leverage must be positive")
        if self.max_leverage > Decimal("10"):  # Conservative leverage limit
            result.add_error("max_leverage cannot exceed 10x")

        # Validate slippage
        if self.max_slippage_pct <= 0:
            result.add_error("max_slippage_pct must be positive")
        if self.max_slippage_pct > Decimal("0.05"):  # 5% max slippage
            result.add_error("max_slippage_pct cannot exceed 5%")

        # Validate timeouts
        if self.position_timeout < 60:  # Minimum 1 minute
            result.add_error("position_timeout must be at least 60 seconds")
        if self.position_timeout > 86400:  # Maximum 24 hours
            result.add_error("position_timeout cannot exceed 24 hours")

        # Validate profit targets
        if self.min_profit_pct <= 0:
            result.add_error("min_profit_pct must be positive")
        if self.min_profit_pct >= Decimal("0.1"):  # 10% minimum profit is unrealistic
            result.add_error("min_profit_pct cannot exceed 10%")

        # Validate correlation limits
        if self.max_correlation <= 0:
            result.add_error("max_correlation must be positive")
        if self.max_correlation > Decimal("1"):
            result.add_error("max_correlation cannot exceed 1.0")

        # Validate open trade limits
        if self.max_open_trades < 1:
            result.add_error("max_open_trades must be at least 1")
        if self.max_open_trades > self.max_positions:
            result.add_error("max_open_trades cannot exceed max_positions")

        # Validate daily loss limits
        if self.max_daily_loss <= 0:
            result.add_error("max_daily_loss must be positive")
        if self.max_daily_loss >= Decimal("0.1"):  # 10% daily loss limit
            result.add_error("max_daily_loss cannot exceed 10%")
        if self.max_daily_loss <= self.emergency_stop_pct:
            result.add_error("max_daily_loss must be greater than emergency_stop_pct")

        # Validate risk ratios
        total_risk = self.max_position_size * self.max_leverage
        if total_risk > Decimal("1"):
            result.add_error(
                "Combined position size and leverage cannot exceed 100% exposure"
            )

        # Validate risk relationships
        if self.emergency_stop_pct >= self.max_daily_loss:
            result.add_error("emergency_stop_pct must be less than max_daily_loss")
        if self.max_daily_loss >= self.max_drawdown:
            result.add_error("max_daily_loss must be less than max_drawdown")

        return result


@dataclass
class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""

    pass


# trading config class that represents the trading config
@dataclass
class TradingConfig:
    """Trading configuration with validation"""

    risk: RiskConfig
    timeframe: str
    market_list: list
    initial_balance: Decimal
    paper_mode: bool = False
    max_trades_per_day: int = 10
    min_trade_interval: int = 60  # seconds
    backtest_mode: bool = False
    debug_mode: bool = False
    dry_run: bool = False
    min_volume: Decimal = Decimal("0")  # Minimum trading volume
    max_spread_pct: Decimal = Decimal("0.02")  # Maximum allowed spread
    market_hours: Dict[str, tuple] = field(
        default_factory=dict
    )  # Trading hours per market
    excluded_days: Set[str] = field(default_factory=set)  # Days to exclude from trading
    max_retry_attempts: int = 3  # Maximum retry attempts for operations
    heartbeat_interval: int = 30  # Heartbeat check interval in seconds
    health_check_interval: int = 60  # Health check interval in seconds
    # Add new balance sync settings
    balance_sync_interval: int = 300  # Balance sync interval in seconds
    max_balance_sync_delay: int = 600  # Maximum allowed delay between balance syncs
    max_balance_deviation_pct: Decimal = Decimal(
        "0.01"
    )  # Maximum allowed balance deviation (1%)
    require_balance_sync: bool = True  # Whether to require balance sync before trading
    max_sync_retries: int = 3  # Maximum balance sync retry attempts
    sync_retry_delay: int = 5  # Delay between sync retries in seconds
    db_validation_timeout: int = 10  # Database validation timeout in seconds
    db_reconnect_attempts: int = 3  # Database reconnection attempts
    db_reconnect_delay: int = 5  # Delay between database reconnection attempts

    @classmethod
    def from_dict(cls, data: Dict) -> ValidationResult:
        """Create config from dictionary with validation"""
        try:
            nh = NumericHandler()

            # Validate and set risk config first
            if "risk" not in data:
                raise ConfigValidationError(
                    "Missing required 'risk' configuration section"
                )
            risk_config = RiskConfig.from_dict(data["risk"])

            # Validate timeframe
            if "timeframe" not in data:
                raise ConfigValidationError(
                    "Missing required 'timeframe' configuration"
                )
            timeframe = str(data["timeframe"]).upper()
            valid_timeframes = {"1M", "5M", "15M", "30M", "1H", "4H", "1D"}
            if timeframe not in valid_timeframes:
                raise ConfigValidationError(
                    f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
                )

            # Validate market list
            if "market_list" not in data:
                raise ConfigValidationError(
                    "Missing required 'market_list' configuration"
                )
            market_list = data["market_list"]
            if not isinstance(market_list, list) or not market_list:
                raise ConfigValidationError("market_list must be a non-empty list")

            # Validate initial balance
            if "initial_balance" not in data:
                raise ConfigValidationError(
                    "Missing required 'initial_balance' configuration"
                )
            try:
                initial_balance = nh.to_decimal(data["initial_balance"])
                if initial_balance <= 0:
                    raise ConfigValidationError("initial_balance must be positive")
            except (ValueError, TypeError) as e:
                raise ConfigValidationError(f"Invalid initial_balance: {e}")

            # Optional parameters with validation
            max_trades = int(data.get("max_trades_per_day", 10))
            if max_trades <= 0:
                raise ConfigValidationError("max_trades_per_day must be positive")

            min_interval = int(data.get("min_trade_interval", 60))
            if min_interval < 1:
                raise ConfigValidationError(
                    "min_trade_interval must be at least 1 second"
                )

            # Additional validations for new fields
            min_volume = nh.to_decimal(data.get("min_volume", "0"))
            max_spread = nh.percentage_to_decimal(data.get("max_spread_pct", "0.02"))

            # Parse market hours if provided
            market_hours = {}
            if "market_hours" in data:
                for market, hours in data["market_hours"].items():
                    try:
                        start, end = hours.split("-")
                        market_hours[market] = (start.strip(), end.strip())
                    except (ValueError, AttributeError):
                        raise ConfigValidationError(
                            f"Invalid market hours format for {market}"
                        )

            # Parse excluded days
            excluded_days = set(data.get("excluded_days", []))
            valid_days = {
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            }
            invalid_days = excluded_days - valid_days
            if invalid_days:
                raise ConfigValidationError(f"Invalid excluded days: {invalid_days}")

            # Additional balance sync validations
            balance_sync_interval = int(data.get("balance_sync_interval", 300))
            max_balance_sync_delay = int(data.get("max_balance_sync_delay", 600))
            max_balance_deviation = nh.percentage_to_decimal(
                data.get("max_balance_deviation_pct", "0.01")
            )
            require_sync = bool(data.get("require_balance_sync", True))
            max_sync_retries = int(data.get("max_sync_retries", 3))
            sync_retry_delay = int(data.get("sync_retry_delay", 5))
            db_timeout = int(data.get("db_validation_timeout", 10))
            db_reconnect_attempts = int(data.get("db_reconnect_attempts", 3))
            db_reconnect_delay = int(data.get("db_reconnect_delay", 5))

            # Create config instance with new fields
            config = cls(
                risk=risk_config,
                timeframe=timeframe,
                market_list=market_list,
                initial_balance=initial_balance,
                paper_mode=bool(data.get("paper_mode", False)),
                max_trades_per_day=max_trades,
                min_trade_interval=min_interval,
                backtest_mode=bool(data.get("backtest_mode", False)),
                debug_mode=bool(data.get("debug_mode", False)),
                dry_run=bool(data.get("dry_run", False)),
                min_volume=min_volume,
                max_spread_pct=max_spread,
                market_hours=market_hours,
                excluded_days=excluded_days,
                max_retry_attempts=int(data.get("max_retry_attempts", 3)),
                heartbeat_interval=int(data.get("heartbeat_interval", 30)),
                health_check_interval=int(data.get("health_check_interval", 60)),
                balance_sync_interval=balance_sync_interval,
                max_balance_sync_delay=max_balance_sync_delay,
                max_balance_deviation_pct=max_balance_deviation,
                require_balance_sync=require_sync,
                max_sync_retries=max_sync_retries,
                sync_retry_delay=sync_retry_delay,
                db_validation_timeout=db_timeout,
                db_reconnect_attempts=db_reconnect_attempts,
                db_reconnect_delay=db_reconnect_delay,
            )

            # Validate the complete config
            config.validate()
            return ValidationResult(is_valid=True)

        except ConfigValidationError:
            raise
        except Exception as e:
            return ValidationResult(is_valid=False, error_message=str(e))

    def validate(self) -> None:
        """Validate the complete configuration"""
        # Validate risk config
        risk_validation = self.risk.validate()
        if not risk_validation.is_valid:
            raise ConfigValidationError(
                f"Risk config validation failed: {risk_validation.error_message}"
            )

        # Additional validation logic
        if self.backtest_mode and not self.paper_mode:
            raise ConfigValidationError(
                "backtest_mode requires paper_mode to be enabled"
            )

        if self.dry_run and not self.paper_mode:
            raise ConfigValidationError("dry_run requires paper_mode to be enabled")

        # Market-specific validation
        for market in self.market_list:
            if not isinstance(market, str) or "/" not in market:
                raise ConfigValidationError(
                    f"Invalid market format: {market}. Expected format: BASE/QUOTE"
                )

        # Validate volume and spread constraints
        if self.min_volume < 0:
            raise ConfigValidationError("min_volume cannot be negative")

        if self.max_spread_pct <= 0 or self.max_spread_pct > Decimal("0.1"):
            raise ConfigValidationError("max_spread_pct must be between 0 and 10%")

        # Validate intervals
        if self.heartbeat_interval < 1:
            raise ConfigValidationError("heartbeat_interval must be positive")
        if self.health_check_interval < self.heartbeat_interval:
            raise ConfigValidationError(
                "health_check_interval must be greater than heartbeat_interval"
            )

        # Validate retry attempts
        if self.max_retry_attempts < 1:
            raise ConfigValidationError("max_retry_attempts must be at least 1")
        if self.max_retry_attempts > 10:
            raise ConfigValidationError("max_retry_attempts cannot exceed 10")

        # Validate market hours format if provided
        for market, (start, end) in self.market_hours.items():
            try:
                # Validate time format (HH:MM)
                for time_str in (start, end):
                    hour, minute = map(int, time_str.split(":"))
                    if not (0 <= hour <= 23 and 0 <= minute <= 59):
                        raise ValueError
            except ValueError:
                raise ConfigValidationError(
                    f"Invalid market hours for {market}. Format should be HH:MM-HH:MM"
                )

        # Validate market list against market hours
        if self.market_hours:
            unknown_markets = set(self.market_hours.keys()) - set(self.market_list)
            if unknown_markets:
                raise ConfigValidationError(
                    f"Market hours specified for unknown markets: {unknown_markets}"
                )

        # Validate balance sync settings
        if self.balance_sync_interval <= 0:
            raise ConfigValidationError("balance_sync_interval must be positive")

        if self.max_balance_sync_delay < self.balance_sync_interval:
            raise ConfigValidationError(
                "max_balance_sync_delay must be greater than balance_sync_interval"
            )

        if self.max_balance_sync_delay > 3600:  # 1 hour max
            raise ConfigValidationError("max_balance_sync_delay cannot exceed 1 hour")

        if self.max_balance_deviation_pct <= 0:
            raise ConfigValidationError("max_balance_deviation_pct must be positive")

        if self.max_balance_deviation_pct > Decimal("0.05"):  # 5% max deviation
            raise ConfigValidationError("max_balance_deviation_pct cannot exceed 5%")

        if self.max_sync_retries < 1:
            raise ConfigValidationError("max_sync_retries must be at least 1")

        if self.max_sync_retries > 10:
            raise ConfigValidationError("max_sync_retries cannot exceed 10")

        if self.sync_retry_delay < 1:
            raise ConfigValidationError("sync_retry_delay must be at least 1 second")

        if self.sync_retry_delay > 60:
            raise ConfigValidationError("sync_retry_delay cannot exceed 60 seconds")

        if self.db_validation_timeout < 1:
            raise ConfigValidationError(
                "db_validation_timeout must be at least 1 second"
            )

        if self.db_validation_timeout > 30:
            raise ConfigValidationError(
                "db_validation_timeout cannot exceed 30 seconds"
            )

        if self.db_reconnect_attempts < 1:
            raise ConfigValidationError("db_reconnect_attempts must be at least 1")

        if self.db_reconnect_attempts > 10:
            raise ConfigValidationError("db_reconnect_attempts cannot exceed 10")

        if self.db_reconnect_delay < 1:
            raise ConfigValidationError("db_reconnect_delay must be at least 1 second")

        if self.db_reconnect_delay > 60:
            raise ConfigValidationError("db_reconnect_delay cannot exceed 60 seconds")

        # Validate balance sync requirements for non-paper trading
        if not self.paper_mode and not self.require_balance_sync:
            raise ConfigValidationError(
                "require_balance_sync must be enabled for non-paper trading"
            )

        # Validate database settings for non-paper trading
        if not self.paper_mode:
            if self.balance_sync_interval > 600:  # 10 minutes max for live trading
                raise ConfigValidationError(
                    "balance_sync_interval cannot exceed 10 minutes for live trading"
                )
            if self.max_balance_deviation_pct > Decimal(
                "0.02"
            ):  # 2% max for live trading
                raise ConfigValidationError(
                    "max_balance_deviation_pct cannot exceed 2% for live trading"
                )
            if self.db_validation_timeout > 15:  # 15 seconds max for live trading
                raise ConfigValidationError(
                    "db_validation_timeout cannot exceed 15 seconds for live trading"
                )

        # Validate initial balance against risk limits
        min_required_balance = Decimal("100")  # Minimum viable trading balance
        if self.initial_balance < min_required_balance and not self.paper_mode:
            raise ConfigValidationError(
                f"initial_balance must be at least {min_required_balance} for live trading"
            )

        # Validate timeframe compatibility
        if self.timeframe == "1M" and not self.paper_mode:
            raise ConfigValidationError("1M timeframe is too short for live trading")

        # Validate market list size
        if len(self.market_list) > 50:  # Reasonable limit for monitoring
            raise ConfigValidationError(
                "Cannot monitor more than 50 markets simultaneously"
            )

        # Validate trade interval relationships
        min_interval_by_timeframe = {
            "1M": 60,  # 1 minute
            "5M": 300,  # 5 minutes
            "15M": 900,  # 15 minutes
            "30M": 1800,  # 30 minutes
            "1H": 3600,  # 1 hour
            "4H": 14400,  # 4 hours
            "1D": 86400,  # 1 day
        }
        if self.min_trade_interval < min_interval_by_timeframe[self.timeframe]:
            raise ConfigValidationError(
                f"min_trade_interval must be at least {min_interval_by_timeframe[self.timeframe]} seconds for {self.timeframe} timeframe"
            )

        # Validate health check relationships
        if self.health_check_interval >= self.balance_sync_interval:
            raise ConfigValidationError(
                "health_check_interval must be less than balance_sync_interval"
            )

        # Validate paper/live mode specific requirements
        if not self.paper_mode:
            # Validate market hours coverage
            if not self.market_hours and len(self.market_list) > 0:
                raise ConfigValidationError(
                    "market_hours must be specified for live trading"
                )

            # Validate retry settings for live trading
            if self.max_retry_attempts < 2:
                raise ConfigValidationError(
                    "max_retry_attempts must be at least 2 for live trading"
                )

            # Validate balance sync for different timeframes
            max_sync_intervals = {
                "1M": 60,  # 1 minute
                "5M": 300,  # 5 minutes
                "15M": 900,  # 15 minutes
                "30M": 1800,  # 30 minutes
                "1H": 3600,  # 1 hour
                "4H": 3600,  # Still 1 hour
                "1D": 3600,  # Still 1 hour
            }
            if self.balance_sync_interval > max_sync_intervals[self.timeframe]:
                raise ConfigValidationError(
                    f"balance_sync_interval cannot exceed {max_sync_intervals[self.timeframe]} seconds for {self.timeframe} timeframe in live trading"
                )

        # Validate database settings based on trading volume
        if self.max_trades_per_day > 100 and self.db_validation_timeout < 5:
            raise ConfigValidationError(
                "db_validation_timeout must be at least 5 seconds for high-frequency trading"
            )

        # Validate emergency procedures
        if not self.paper_mode:
            if self.max_balance_deviation_pct > self.risk.emergency_stop_pct:
                raise ConfigValidationError(
                    "max_balance_deviation_pct cannot exceed emergency_stop_pct"
                )
            if self.balance_sync_interval > (self.risk.position_timeout / 10):
                raise ConfigValidationError(
                    "balance_sync_interval must be at most 10% of position_timeout"
                )

        # Validate concurrent operation limits
        max_concurrent_markets = 20 if not self.paper_mode else 50
        if len(self.market_list) > max_concurrent_markets:
            raise ConfigValidationError(
                f"Cannot monitor more than {max_concurrent_markets} markets in {'live' if not self.paper_mode else 'paper'} mode"
            )

        # Validate system resource constraints
        if self.heartbeat_interval < 5 and len(self.market_list) > 10:
            raise ConfigValidationError(
                "heartbeat_interval must be at least 5 seconds when monitoring more than 10 markets"
            )


# config loader class that loads the config
class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.nh = NumericHandler()
        self._config: Optional[TradingConfig] = None
        self._last_modified: Optional[float] = None
        self._load_attempts = 0
        self.MAX_LOAD_ATTEMPTS = 3

    def load_config(self) -> TradingConfig:
        """Load and validate configuration with retry logic"""
        try:
            self._load_attempts += 1

            # Check if config file exists
            if not self.config_path.exists():
                raise ConfigValidationError(
                    f"Config file not found: {self.config_path}"
                )

            # Check file modification time
            current_mtime = self.config_path.stat().st_mtime
            if self._config and self._last_modified == current_mtime:
                return self._config

            # Load config data
            try:
                if self.config_path.suffix == ".yaml":
                    with open(self.config_path) as f:
                        config_data = yaml.safe_load(f)
                else:
                    with open(self.config_path) as f:
                        config_data = json.load(f)
            except Exception as e:
                raise ConfigValidationError(f"Failed to parse config file: {e}")

            if not isinstance(config_data, dict):
                raise ConfigValidationError(
                    "Config file must contain a JSON/YAML object"
                )

            # Create and validate config
            self._config = TradingConfig.from_dict(config_data)
            self._last_modified = current_mtime
            self._load_attempts = 0
            return self._config

        except ConfigValidationError:
            # Don't retry on validation errors
            raise
        except Exception as e:
            if self._load_attempts >= self.MAX_LOAD_ATTEMPTS:
                raise ConfigValidationError(
                    f"Failed to load config after {self.MAX_LOAD_ATTEMPTS} attempts: {e}"
                )
            # Retry with exponential backoff
            time.sleep(2 ** (self._load_attempts - 1))
            return self.load_config()

    def get_config(self) -> TradingConfig:
        """Get current configuration or load if not loaded"""
        if self._config is None:
            return self.load_config()
        return self._config

    def reload_config(self) -> TradingConfig:
        """Force reload configuration from file"""
        self._config = None
        self._last_modified = None
        self._load_attempts = 0
        return self.load_config()
