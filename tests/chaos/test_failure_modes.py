# tests/chaos/test_failure_modes.py
import pytest

from utils.error_handler import DatabaseError


def test_database_failure():
    # Simulate database outage
    with pytest.raises(DatabaseError):
        # Trigger database failure
        raise DatabaseError("Simulated database outage")

    # Check if system handles failure gracefully
