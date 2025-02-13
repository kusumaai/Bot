from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ExchangeStatus:
    # Required fields (no defaults)
    exchange_id: str
    status: str
    timestamp: float
    api_status: str
    trading_status: str
    
    # Optional fields (with defaults)
    latency_ms: float = 0.0
    rate_limits: Dict[str, int] = field(default_factory=dict)
    maintenance_mode: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict) 