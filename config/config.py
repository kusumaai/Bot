import os
import json
from typing import Dict, Any
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    try:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            
        # Load and parse config file
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Ensure required config sections exist
        required_sections = [
            "market_list",
            "timeframe",
            "execution_interval",
            "trading_fees",
            "paper_mode"
        ]
        
        missing = [section for section in required_sections if section not in config]
        if missing:
            raise KeyError(f"Missing required configuration sections: {', '.join(missing)}")
            
        return config
        
    except Exception as e:
        raise Exception(f"Failed to load configuration: {str(e)}") 