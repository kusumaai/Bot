#! /usr/bin/env python3
#src/monitoring/alerts.py
"""
Module: src.monitoring
Provides alert management.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List
#alert config class that defines the alert configuration
@dataclass
class AlertConfig:
    # Required fields (no defaults)
    alert_type: str
    severity: str
    threshold: Decimal
    message_template: str
    
    # Optional fields (with defaults)
    enabled: bool = True
    cooldown_minutes: int = 60
    recipients: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) 
 
 #alert class that defines the alert
class Alert:
    def __init__(self, config: AlertConfig):
        self.config = config
        
    def send(self):
        pass
    
#alert manager class that manages alerts
class AlertManager:
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alerts = []
        
    def add_alert(self, alert: Alert):
        self.alerts.append(alert)
        
    def send_alerts(self):
        for alert in self.alerts:
            alert.send()    
            
#alert types
class AlertType:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def send(self, alert: Alert):
        pass
    
