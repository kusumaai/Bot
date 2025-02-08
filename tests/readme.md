# tests/README.md
"""
KillaBot Test Suite
==================

This directory contains the test suite for the KillaBot trading system.

Directory Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- validation_tests.py: Comprehensive validation suite
- conftest.py: PyTest configuration and shared fixtures

Running Tests:
-------------
From project root:
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/validation_tests.py

# Run with verbose output
pytest -v tests/
```

Writing Tests:
-------------
1. Unit tests go in tests/unit/
2. Integration tests go in tests/integration/
3. Use fixtures from conftest.py for shared resources
4. Follow existing test patterns for consistency