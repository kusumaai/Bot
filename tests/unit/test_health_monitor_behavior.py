#! /usr/bin/env python3
# tests/unit/test_health_monitor_behavior.py
"""
Module: tests.unit
Provides unit testing functionality for the health monitor behavior module.
"""
import asyncio

import pytest

from utils.health_monitor import HealthMonitor


# Create fake objects to simulate health monitor conditions
class FakeDBQueries:
    async def ping(self):
        return True


class FakeExchangeInterface:
    class FakeExchange:
        async def ping(self):
            return False  # Simulate exchange failure

    exchange = FakeExchange()


@pytest.fixture
def logger():
    import logging

    return logging.getLogger("test_health")


@pytest.fixture
def health_monitor():
    class FakeContext:
        db_queries = FakeDBQueries()
        exchange_interface = FakeExchangeInterface()
        logger = logger()

    return HealthMonitor(FakeContext())


@pytest.mark.asyncio
async def test_health_monitor_exchange_failure(health_monitor):
    with pytest.raises(Exception):
        # Assuming check_exchange is a method that pings the exchange and raises on failure
        await health_monitor.check_exchange()
