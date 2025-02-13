from typing import Any, Dict, Optional, List
import asyncio
from decimal import Decimal
import logging
import time

from trading.exceptions import ExchangeAPIError

class PaperExchange:
    def __init__(self):
        self.positions = {}
        self.balances = {"USDT": 10000}  # Default paper balance
        self.orders = {}
        
    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None):
        order_id = str(len(self.orders) + 1)
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "closed"
        }
        self.orders[order_id] = order
        return order

    async def fetch_balance(self):
        return self.balances

    async def fetch_positions(self):
        return list(self.positions.values())

    async def close_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        order = self.orders.pop(order_id, None)
        if order:
            order['status'] = 'CLOSED'
            return order
        return None

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        return self.orders.get(order_id)

    async def fetch_markets(self) -> Dict[str, Any]:
        # Simulate fetching markets
        markets = {
            'BTC/USDT': {
                'symbol': 'BTC/USDT',
                'base': 'BTC',
                'quote': 'USDT',
                'active': True
            },
            'ETH/USDT': {
                'symbol': 'ETH/USDT',
                'base': 'ETH',
                'quote': 'USDT',
                'active': True
            }
            # Add more simulated markets as needed
        }
        return markets

    async def load_markets(self) -> Dict[str, Any]:
        return await self.fetch_markets()

    async def fetch_candles(self, symbol: str, timeframe: str, limit: int) -> Optional[List[Dict[str, Any]]]:
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

    async def ping(self) -> bool:
        # Simulate a ping to the exchange
        await asyncio.sleep(0.1)  # Simulate network delay
        return True

    async def close(self):
        pass;
    