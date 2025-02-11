from decimal import Decimal
from typing import Dict, Any, Optional
import time
from dataclasses import dataclass

@dataclass
class MarketDataValidation:
    timestamp: float
    symbol: str
    price: Decimal
    volume: Decimal
    is_valid: bool
    error_message: Optional[str] = None

def validate_market_data(data: Dict[str, Any]) -> MarketDataValidation:
    try:
        timestamp = float(data.get("timestamp", 0))
        if time.time() - timestamp > 5:  # 5 second staleness check
            return MarketDataValidation(
                timestamp=timestamp,
                symbol=data.get("symbol", ""),
                price=Decimal(0),
                volume=Decimal(0),
                is_valid=False,
                error_message="Stale data"
            )
        return MarketDataValidation(
            timestamp=timestamp,
            symbol=data.get("symbol", ""),
            price=Decimal(str(data.get("price", 0))),
            volume=Decimal(str(data.get("volume", 0))),
            is_valid=True
        )
    except Exception as e:
        return MarketDataValidation(
            timestamp=0,
            symbol=data.get("symbol", ""),
            price=Decimal(0),
            volume=Decimal(0),
            is_valid=False,
            error_message=str(e)
        )
