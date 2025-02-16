# KillaBot 0.9

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

1. **Clone the Repository:**

    \`\`\`bash
    git clone <https://github.com/yourusername/KillaBot.git>
    cd KillaBot
    \`\`\`

2. **Set Up a Virtual Environment:**

    \`\`\`bash
    python -m venv venv
    source venv/bin/activate  # Linux/MacOS
    venv\Scripts\activate     # Windows
    \`\`\`

3. **Install Dependencies:**

    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

4. **(Optional) Install Development Dependencies via Poetry:**

    \`\`\`bash
    poetry install
    \`\`\`

## Configuration

Adjust the parameters in `src/config/config.json` to tailor KillaBot to your trading style. Key configurable parameters include:

- **Trading Mode:** Toggle between paper trading and live trading.
- **Market Settings:** Define market pairs, timeframes, and trading intervals.
- **Risk Parameters:** Set limits for position size, leverage, drawdown, daily loss, and emergency stop thresholds.
- **Ratchet Settings:** Configure thresholds and lock-in percentages for trailing stop management.
- **Exchange Settings:** Specify connection settings such as timeouts and rate limits.

The risk configuration is further validated and managed in `src/config/risk_config.py`, ensuring that all parameters comply with pre-set risk constraints.

## Testing

Run the test suite:

\`\`\`bash

### Run all tests

pytest tests/

### Run specific tests

pytest tests/test_risk_manager.py
pytest tests/validation_tests.py

### Run with verbose output

pytest -v tests/
\`\`\`

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

### Monitor logs

\`\`\`bash
tail -f logs/trading.log  # Main trading log
tail -f logs/error.log   # Error log
\`\`\`

## Error Handling

The system uses a centralized error handling system:

\`\`\`python
try:
    # Trading operations
except Exception as e:
    handle_error(e, "component.function_name", logger=logger)
\`\`\`

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

## CI/CD Integration

1. **GitHub Actions**

2. **Async Testing**

## Usage

Launch the bot via the command line:

\`\`\`bash
python -m src.execution.bot
\`\`\`

**Component Initialization Sequence:**

1. **Database:** Establishes the SQLite connection and initializes the schema.
2. **Risk & Portfolio Managers:** Sets up real-time risk controls and portfolio monitoring.
3. **Exchange & Market Data:** Connects to all configured exchanges for live market data.
4. **Circuit Breaker & Health Monitor:** Continuously monitors system integrity.
5. **Ratchet Manager:** Applies dynamic stop adjustments as market conditions evolve.
6. **Main Loop:** Enters the asynchronous trading loop, executing strategies and managing orders.

## Unit Testing

KillaBot comes with a robust test suite structured as follows:

- **Unit Tests:** Located in `tests/unit/`
- **Integration Tests:** Located in `tests/integration/`
- **Validation Tests:** A comprehensive suite found in `tests/validation_tests.py`

Run all tests with:

\`\`\`bash
pytest tests/
\`\`\`

For specific sections, run:

\`\`\`bash
pytest tests/test_risk_manager.py
pytest tests/validation_tests.py
\`\`\`

---

## Development & CI/CD

- **Code Quality:** The project adheres to Black formatting, Flake8 linting, and MyPy type checking to maintain high code quality.
- **Pre-commit Hooks:** Ensure consistency and catch issues early in the development cycle.
- **Continuous Integration:** Integrated GitHub Actions tests the codebase on every push, ensuring that all changes meet stringent quality standards.
- **Extensibility:** The modular design allows for easy additions of new signal generators, risk modules, or other enhancements.

---

## Monitoring, Logging & Error Handling

- **Logging:** Configurable logging outputs are maintained both on the console and in log files (e.g., `logs/trading_bot.log`, `logs/error.log`).
- **Health Checks:** The Health Monitor module collects system metrics, ensuring the trading engine remains within optimal performance parameters.
- **Error Handling:** A centralized error handler captures and logs exceptions, ensuring graceful recovery and continuity during unexpected events.

---

## Backtesting

The backtesting framework (found in `src/backtesting/backtester.py`) allows you to evaluate your strategies using historical data. Key aspects include:

- **Simulation:** Run strategy simulations over custom date ranges.
- **Analytics:** Generate detailed trade statistics (PnL, win/loss ratios, fees paid, etc.).
- **Strategy Evaluation:** Integrates both ML signals and genetic algorithm optimizations for continuous strategy improvement.

---

## Acknowledgements

Developed by Phil Sanderson and Mike Van-Dijk

---

## Contact

For support or inquiries, please contact <philsanderson@pm.me>

---

Trade smart and safe. Destroy the market, with KillaBot!
