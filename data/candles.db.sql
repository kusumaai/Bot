BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "account" (
	"id"	INTEGER,
	"exchange"	TEXT NOT NULL,
	"balance"	REAL NOT NULL,
	"used_balance"	REAL NOT NULL,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "bot_performance" (
	"day"	TEXT,
	"real_trades_closed"	INTEGER,
	"paper_trades_closed"	INTEGER,
	"real_pnl"	REAL,
	"paper_pnl"	REAL,
	PRIMARY KEY("day")
);
CREATE TABLE IF NOT EXISTS "candles" (
	"symbol"	TEXT,
	"timeframe"	TEXT,
	"timestamp"	INTEGER,
	"open"	REAL,
	"high"	REAL,
	"low"	REAL,
	"close"	REAL,
	"volume"	REAL,
	"datetime"	TEXT,
	"atr_14"	REAL,
	"exchange"	TEXT
);
CREATE TABLE IF NOT EXISTS "ga_rules" (
	"id"	TEXT,
	"chromosome_json"	TEXT,
	"fitness"	REAL,
	"date_created"	TEXT,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "sentiment_features" (
	"date"	TEXT,
	"fng_sentiment"	REAL,
	"btc_dominance"	REAL,
	"usdt_dominance"	REAL,
	"usdc_dominance"	REAL,
	PRIMARY KEY("date")
);
CREATE TABLE IF NOT EXISTS "supported_pairs" (
	"exchange"	TEXT NOT NULL,
	"symbol"	TEXT NOT NULL,
	"supported"	INTEGER NOT NULL DEFAULT 1,
	"last_checked"	DATETIME NOT NULL,
	PRIMARY KEY("exchange","symbol")
);
CREATE TABLE IF NOT EXISTS "trades" (
	"id"	TEXT,
	"symbol"	TEXT NOT NULL,
	"timeframe"	TEXT,
	"trade_source"	TEXT,
	"direction"	TEXT,
	"entry_price"	REAL,
	"sl"	REAL,
	"tp"	REAL,
	"entry_time"	TEXT,
	"close_time"	TEXT,
	"result"	REAL,
	"close_reason"	TEXT,
	"exchange"	TEXT,
	"position_size"	REAL,
	PRIMARY KEY("id")
);
COMMIT;
