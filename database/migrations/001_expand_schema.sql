-- Start transaction for atomic migration
BEGIN TRANSACTION;

-- Backup existing candles table just in case
CREATE TABLE IF NOT EXISTS candles_backup AS SELECT * FROM candles;

-- Add indexes to existing candles table (will fail silently if they exist)
CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe ON candles(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp);

-- Create new tables with proper constraints
CREATE TABLE IF NOT EXISTS account_balance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance DECIMAL NOT NULL,
    timestamp INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USDT'
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size DECIMAL NOT NULL,
    entry_price DECIMAL NOT NULL,
    status TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    take_profit DECIMAL,
    stop_loss DECIMAL,
    metadata TEXT,
    CONSTRAINT valid_status CHECK (status IN ('open', 'closed'))
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_timestamp ON positions(timestamp);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size DECIMAL NOT NULL,
    price DECIMAL NOT NULL,
    timestamp INTEGER NOT NULL,
    type TEXT NOT NULL,
    pnl DECIMAL,
    fees DECIMAL,
    metadata TEXT,
    FOREIGN KEY(position_id) REFERENCES positions(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id);

CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day DATE NOT NULL UNIQUE,
    real_trades_closed INTEGER DEFAULT 0,
    paper_trades_closed INTEGER DEFAULT 0,
    real_pnl DECIMAL DEFAULT 0,
    paper_pnl DECIMAL DEFAULT 0,
    max_drawdown DECIMAL,
    sharpe_ratio REAL,
    win_rate REAL
);

CREATE INDEX IF NOT EXISTS idx_performance_day ON performance(day);

CREATE TABLE IF NOT EXISTS error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    stack_trace TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_error_log_timestamp ON error_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_error_log_type ON error_log(error_type);

-- Insert initial account balance if none exists
INSERT OR IGNORE INTO account_balance (balance, timestamp, currency)
SELECT 10000.0, unixepoch(), 'USDT'
WHERE NOT EXISTS (SELECT 1 FROM account_balance LIMIT 1);

-- Verify migration success
SELECT CASE 
    WHEN EXISTS (SELECT 1 FROM candles_backup) 
    AND EXISTS (SELECT 1 FROM account_balance)
    AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_candles_symbol_timeframe')
    THEN 'Migration successful'
    ELSE 'Migration failed'
END as migration_status;

-- If everything succeeded, commit the transaction
COMMIT; 