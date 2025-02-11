from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Any

class IPortfolio(ABC):
    @abstractmethod
    def calculate_value(self) -> Decimal:
        pass
        
    @abstractmethod
    def calculate_drawdown(self) -> Decimal:
        pass 