from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List


@dataclass
class AlertConfig:
    # Required fields (no defaults)
    alert_type: str
    severity: str
    threshold: Decimal
    message_template: str
    
    # Optional fields (with defaults)
    enabled: bool = True
    cooldown_minutes: int = 60
    recipients: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) 