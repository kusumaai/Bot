from typing import Any, Dict, Optional, List
import asyncio
from decimal import Decimal
import logging
import time

class PaperExchange:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str]):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.orders: Dict[str, Dict[str, Any]] = {}

    async def create_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        order_id = f"paper_{int(time.time())}"
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'amount': str(Decimal(amount)),
            'price': str(Decimal(price)) if price else None,
            'status': 'OPEN'
        }
        self.orders[order_id] = order
        self.logger.info(f"Paper order created: {order}")
        return order

    async def close_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        order = self.orders.pop(order_id, None)
        if order:
            order['status'] = 'CLOSED'
            self.logger.info(f"Paper order closed: {order}")
            return order
        return None

    async def get_candles(self, symbol: str, timeframe: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        # Simulate fetching candles
        candles = []
        current_time = int(time.time())
        for i in range(limit):
            candle = {
                'timestamp': current_time - i * 60,
                'open': '50000',
                'high': '51000',
                'low': '49000',
                'close': '50500',
                'volume': '100'
            }
            candles.append(candle)
        return candles

    async def close(self):
        # Cleanup if necessary
        self.logger.info("PaperExchange connection closed.") 