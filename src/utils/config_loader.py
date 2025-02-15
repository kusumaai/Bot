#! /usr/bin/env python3
#src/utils/config_loader.py
"""
Module: src.utils
Provides configuration loading functionality.
"""
from decimal import Decimal
from typing import Dict, Any, Optional, Set
import yaml
import json
from pathlib import Path
from dataclasses import dataclass
from utils.numeric import NumericHandler

#risk config class that represents the risk config  
@dataclass
class RiskConfig:
    max_position_size: Decimal
    emergency_stop_pct: Decimal
    max_drawdown: Decimal
    max_daily_loss: Decimal
    max_positions: int
    #from dict method
    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskConfig':
        nh = NumericHandler()
        return cls(
            max_position_size=nh.percentage_to_decimal(data['max_position_size']),
            emergency_stop_pct=nh.percentage_to_decimal(data['emergency_stop_pct']),
            max_drawdown=nh.percentage_to_decimal(data['max_drawdown']),
            max_daily_loss=nh.percentage_to_decimal(data['max_daily_loss']),
            max_positions=int(data['max_positions'])
        )
#trading config class that represents the trading config
@dataclass
class TradingConfig:
    risk: RiskConfig
    timeframe: str
    market_list: list
    initial_balance: Decimal

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingConfig':
        nh = NumericHandler()
        return cls(
            risk=RiskConfig.from_dict(data['risk']),
            timeframe=data['timeframe'],
            market_list=data['market_list'],
            initial_balance=nh.to_decimal(data['initial_balance'])
        )
#config loader class that loads the config  
class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.nh = NumericHandler()
        self._config: Optional[TradingConfig] = None
        
    def load_config(self) -> TradingConfig:
        """Load and validate configuration"""
        try:
            if self.config_path.suffix == '.yaml':
                with open(self.config_path) as f:
                    config_data = yaml.safe_load(f)
            else:
                with open(self.config_path) as f:
                    config_data = json.load(f)
                    
            self._validate_config(config_data)
            self._config = TradingConfig.from_dict(config_data)
            return self._config
            
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")
    #validate  the config
    def _validate_config(self, config: Dict) -> None:
        """Validate configuration values"""
        # Define required fields and their types
        risk_fields: Set[str] = {
            'max_position_size',
            'emergency_stop_pct',
            'max_drawdown',
            'max_daily_loss',
            'max_positions'
        }
        
        required_fields = {
            'risk': risk_fields,
            'timeframe': None,
            'market_list': None,
            'initial_balance': None
        }
        
        # Check required fields exist
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate risk parameters
        risk = config['risk']
        for field in risk_fields:
            if field not in risk:
                raise ValueError(f"Missing risk field: {field}")
        
        # Type validations
        if not isinstance(config['timeframe'], str):
            raise ValueError("timeframe must be a string")
            
        if not isinstance(config['market_list'], list):
            raise ValueError("market_list must be a list")
            
        try:
            self.nh.to_decimal(config['initial_balance'])
        except:
            raise ValueError("initial_balance must be convertible to Decimal")
            
        # Validate numeric ranges
        if self.nh.to_decimal(risk['max_position_size']) > Decimal('0.5'):
            raise ValueError("max_position_size cannot exceed 50%")
            
        if self.nh.to_decimal(risk['max_drawdown']) > Decimal('0.2'):
            raise ValueError("max_drawdown cannot exceed 20%")
            
        # Validate market list
        if not config['market_list']:
            raise ValueError("market_list cannot be empty")
    #get the config
    def get_config(self) -> TradingConfig:
        """Get current configuration"""
        if self._config is None:
            raise ValueError("Configuration not loaded")
        return self._config
    #reload the config
    def reload_config(self) -> TradingConfig:
        """Reload configuration from file"""
        return self.load_config()