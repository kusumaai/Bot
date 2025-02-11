from dataclasses import dataclass
import yaml
from decimal import Decimal

@dataclass
class RiskConfig:
    max_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    emergency_stop_pct: Decimal
    position_timeout_hours: int
    
    @classmethod
    def from_yaml(cls, path: str) -> 'RiskConfig':
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            max_position_size=Decimal(str(config['max_position_size'])),
            max_positions=config['max_positions'],
            max_leverage=Decimal(str(config['max_leverage'])),
            emergency_stop_pct=Decimal(str(config['emergency_stop_pct'])),
            position_timeout_hours=config['position_timeout_hours']
        ) 