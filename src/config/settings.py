#! /usr/bin/env python3
#src/config/settings.py
"""
Module: src.config
Provides configuration management.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
from decimal import Decimal
import yaml
import logging
from dataclasses import dataclass
from bot_types.base_types import ValidationResult
@dataclass
class DatabaseSettings:
    path: str
    pool_size: int = 5
    max_connections: int = 20
    timeout: float = 30.0

@dataclass
class LogSettings:
    level: str = "INFO"
    file_path: Optional[str] = None
    max_size: int = 10_485_760  # 10MB
    backup_count: int = 5

class Settings:
    """Central configuration management"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.logger = logging.getLogger("Settings")
        self._settings: Dict[str, Any] = {}
        self._load_config()
        self._validate_settings()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path) as f:
                self._settings = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {e}")
    
    def _validate_settings(self) -> None:
        """Validate required settings and types"""
        required_settings = {
            'database': {'path': str, 'pool_size': int},
            'logging': {'level': str},
            'timeframe': str,
            'risk_factor': (float, Decimal),
            'emergency_stop_pct': (float, Decimal)
        }
        
        self._validate_structure(required_settings)
    
    def _validate_structure(self, required: Dict[str, Union[Dict, Type, tuple]]) -> None:
        """Recursively validate configuration structure"""
        for key, value_type in required.items():
            if key not in self._settings:
                raise ValueError(f"Missing required setting: {key}")
                
            if isinstance(value_type, dict):
                if not isinstance(self._settings[key], dict):
                    raise TypeError(f"Setting {key} must be a dictionary")
                self._validate_structure(value_type)
            else:
                value = self._settings[key]
                if not isinstance(value, value_type if isinstance(value_type, type) else value_type[0]):
                    raise TypeError(f"Invalid type for {key}: expected {value_type}")
    
    @property
    def database(self) -> DatabaseSettings:
        """Get database settings"""
        db_config = self._settings.get('database', {})
        return DatabaseSettings(
            path=db_config['path'],
            pool_size=db_config.get('pool_size', 5),
            max_connections=db_config.get('max_connections', 20),
            timeout=db_config.get('timeout', 30.0)
        )
    
    @property
    def logging(self) -> LogSettings:
        """Get logging settings"""
        log_config = self._settings.get('logging', {})
        return LogSettings(
            level=log_config.get('level', 'INFO'),
            file_path=log_config.get('file_path'),
            max_size=log_config.get('max_size', 10_485_760),
            backup_count=log_config.get('backup_count', 5)
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a configuration value"""
        return self._settings.get(key, default) 