#!/usr/bin/env python3
"""
Setup script for KillaBot.

This script will:
  1. Create (or reuse) a local Python virtual environment (venv).
  2. Install required Python dependencies (from requirements.txt).
  3. Initialize the SQLite database with the required schema.
     (Missing tables are created without deleting or altering existing ones.)

Usage:
    python setup.py

Note:
    - Typically, 'setup.py' is used for packaging Python libraries.
      Here, we're re-purposing it as a convenience script to bootstrap
      the trading bot environment.
    - File paths are set relative to this script's location.
"""

import os
import sys
import subprocess
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")
DB_PATH = os.path.join(BASE_DIR, "data", "candles.db")

DB_SCHEMA = r"""
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS candles (
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    timestamp   INTEGER NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    datetime    TEXT    NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS trades (
    id            TEXT    PRIMARY KEY,
    symbol        TEXT    NOT NULL,
    timeframe     TEXT,
    trade_source  TEXT,
    direction     TEXT,
    entry_price   REAL,
    sl            REAL,
    tp            REAL,
    entry_time    TEXT,
    close_time    TEXT,
    result        REAL,
    close_reason  TEXT,
    exchange      TEXT,
    position_size REAL
);

CREATE TABLE IF NOT EXISTS sentiment_features (
    date           TEXT    PRIMARY KEY,
    fng_sentiment  REAL,
    btc_dominance  REAL,
    usdt_dominance REAL,
    usdc_dominance REAL
);

CREATE TABLE IF NOT EXISTS account (
    id           INTEGER PRIMARY KEY,
    exchange     TEXT    NOT NULL,
    balance      REAL    NOT NULL,
    used_balance REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS supported_pairs (
    exchange     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    supported    INTEGER NOT NULL DEFAULT 1,
    last_checked DATETIME NOT NULL,
    PRIMARY KEY (exchange, symbol)
);

CREATE TABLE IF NOT EXISTS ga_rules (
    id               TEXT    PRIMARY KEY,
    chromosome_json  TEXT,
    fitness          REAL,
    date_created     TEXT
);

CREATE TABLE IF NOT EXISTS bot_performance (
    day                TEXT    PRIMARY KEY,
    real_trades_closed INTEGER,
    paper_trades_closed INTEGER,
    real_pnl           REAL,
    paper_pnl          REAL
);

COMMIT;
"""

def create_virtualenv():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print(f"> Creating virtual environment in {VENV_DIR} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        except subprocess.CalledProcessError as exc:
            print(f"Error: Failed to create virtual environment: {exc}")
            sys.exit(1)
    else:
        print(f"> Virtual environment already exists at {VENV_DIR}.")

def install_requirements():
    """Install Python dependencies using pip inside the venv."""
    print("> Installing requirements...")
    pip_executable = os.path.join(VENV_DIR, "bin", "pip")
    if os.name == "nt":  # Windows
        pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")

    if not os.path.exists(pip_executable):
        print("Error: Could not find pip in the virtual environment.")
        sys.exit(1)

    if not os.path.isfile(REQUIREMENTS_FILE):
        print("Warning: No requirements.txt found, skipping package installation.")
        return

    try:
        subprocess.check_call([pip_executable, "install", "-r", REQUIREMENTS_FILE])
    except subprocess.CalledProcessError as exc:
        print(f"Error: Failed to install requirements: {exc}")
        sys.exit(1)

def initialize_database():
    """Create or update the SQLite database with missing tables if they don't exist."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    print(f"> Initializing database at {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(DB_SCHEMA)
        conn.commit()
        print("> Database schema created or confirmed.")
    except sqlite3.Error as e:
        print(f"Error: Failed to initialize database schema: {e}")
        sys.exit(1)
    finally:
        conn.close()

def main():
    print("=== KillaBot Setup ===")
    create_virtualenv()
    install_requirements()
    initialize_database()
    print("Setup completed successfully.\n")
    print("To activate the virtual environment (Linux/macOS):")
    print(f"    source {os.path.join(VENV_DIR, 'bin', 'activate')}")
    print("Or on Windows:")
    print(f"    {os.path.join(VENV_DIR, 'Scripts', 'activate.bat')} (cmd.exe)")
    print("    or")
    print(f"    {os.path.join(VENV_DIR, 'Scripts', 'Activate.ps1')} (PowerShell)")

if __name__ == "__main__":
    main()
