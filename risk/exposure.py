from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict


@dataclass
class ExposureMetrics:
    # Required fields (no defaults)
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    leverage: Decimal
    
    # Optional fields (with defaults)
    long_exposure: Decimal = Decimal('0')
    short_exposure: Decimal = Decimal('0')
    sector_exposure: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict) 