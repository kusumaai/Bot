# Bot 0.9

A robust automated trading system with advanced safety features, risk management, and data validation.

## Architecture

## Core Features

- **Risk Management**
  - Position sizing with Kelly criterion and risk factor scaling
  - Ratcheting stop-loss system with multiple thresholds
  - Portfolio correlation monitoring and limits
  - Circuit breaker system with health monitoring
  - Maximum drawdown and daily loss controls

- **Trading Engine**
  - ML and GA signal generation with synergy scoring
  - Multi-market position management
  - Asynchronous execution with error recovery
  - Real-time position and performance tracking
  - Comprehensive validation checks

## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```       

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system uses configuration files for risk parameters and trading settings:

```python
# Risk configuration example
risk_limits = RiskLimits(
    max_position_size=Decimal("0.1"),
    max_positions=3,
    max_leverage=Decimal("2.0"),
    emergency_stop_pct=Decimal("-3"),
    max_correlation=Decimal("0.7"),
    max_drawdown=Decimal("0.1"),
    max_daily_loss=Decimal("0.03"),
    kelly_scaling=Decimal("0.5"),
    risk_factor=Decimal("0.1")
)
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_risk_manager.py
pytest tests/validation_tests.py

# Run with verbose output
pytest -v tests/
```

The test suite includes:
- Risk management system tests
- Trading validation tests
- Circuit breaker validation
- Position management tests

## Monitoring

The system includes comprehensive monitoring:

- Health monitoring with component status tracking
- Circuit breaker state monitoring
- Performance metrics and risk limit tracking
- Error tracking and logging
- System resource monitoring

Monitor logs:
```bash
tail -f logs/trading.log  # Main trading log
tail -f logs/error.log   # Error log
```

## Error Handling

The system uses a centralized error handling system:
```python
try:
    # Trading operations
except Exception as e:
    handle_error(e, "component.function_name", logger=logger)
```

## Development

1. **Code Style**
- Black formatting
- Flake8 linting
- MyPy type checking
- Pre-commit hooks

2. **Testing**
- Unit tests
- Integration tests
- Coverage reports

3. **Documentation**
- Code documentation
- API documentation
- Testing documentation

## Safety Features

1. **Circuit Breaker**
- Maximum drawdown protection
- Daily loss limits
- Trade frequency limits
- Cooldown periods

2. **Risk Management**
- Position sizing rules
- Correlation limits
- Volatility checks
- Emergency stops

3. **Data Validation**
- Real-time quality checks
- Historical data validation
- Signal verification
- Market condition monitoring

## Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors and contributors are not responsible for any financial losses incurred while using this system.

## License

MIT License - see LICENSE file for details.

## CI/CD Integration

1. **GitHub Actions** 

3. **Async Testing** 