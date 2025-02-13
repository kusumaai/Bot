from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
from datetime import datetime

@dataclass
class CorrelationMetrics:
    # Required fields (no defaults)
    symbol_pair: Tuple[str, str]
    correlation: float
    significance: float
    sample_size: int
    
    # Optional fields (with defaults)
    lookback_days: int = 30
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict) 