#!/usr/bin/env python3
"""
Broker Integrations Module for Premium AI Signals
Supports multiple broker APIs for automated trading execution.
"""

import json
import sqlite3
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import hashlib
import hmac
import base64

logger = logging.getLogger(__name__)

class BrokerBase(ABC):
    """Base class for broker integrations"""
    
    def __init__(self, api_key: str, api_secret: str, demo_mode: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.demo_mode = demo_mode
        self.connected = False
        self.account_balance = 0.0
        self.positions = []
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, direction: str, amount: float, expiry_minutes: int = 1) -> Dict:
        """Place binary options order"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        pass
    
    @abstractmethod
    def close_position(self, position_id: str) -> Dict:
        """Close position"""
        pass


class AlpacaBroker(BrokerBase):
    """Alpaca Trading API integration"""
    
    def __init__(self, api_key: str, api_secret: str, demo_mode: bool = True):
        super().__init__(api_key, api_secret, demo_mode)
        self.base_url = "https://paper-api.alpaca.markets" if demo_mode else "https://api.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json"
        }
    
    def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            if response.status_code == 200:
                self.connected = True
                account_data = response.json()
                self.account_balance = float(account_data.get('cash', 0))
                logger.info(f"Connected to Alpaca {'Paper' if self.demo_mode else 'Live'} account")
                return True
            else:
                logger.error(f"Alpaca connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Alpaca connection error: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get Alpaca account information"""
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Error getting Alpaca account info: {e}")
            return {}
    
    def place_order(self, symbol: str, direction: str, amount: float, expiry_minutes: int = 1) -> Dict:
        """Place order on Alpaca (simulated binary options)"""
        try:
            # Convert binary options to stock orders for simulation
            side = "buy" if direction == "CALL" else "sell"
            qty = int(amount / 10)  # Convert amount to shares
            
            order_data = {
                "symbol": symbol.replace("USD", ""),  # Convert EURUSD to EUR
                "qty": qty,
                "side": side,
                "type": "market",
                "time_in_force": "day"
            }
            
            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data
            )
            
            if response.status_code == 201:
                order = response.json()
                return {
                    'success': True,
                    'order_id': order['id'],
                    'symbol': symbol,
                    'direction': direction,
                    'amount': amount,
                    'status': 'placed',
                    'broker': 'alpaca'
                }
            else:
                return {
                    'success': False,
                    'error': f"Order failed: {response.status_code}",
                    'broker': 'alpaca'
                }
                
        except Exception as e:
            logger.error(f"Alpaca order error: {e}")
            return {
                'success': False,
                'error': str(e),
                'broker': 'alpaca'
            }
    
    def get_positions(self) -> List[Dict]:
        """Get Alpaca positions"""
        try:
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {e}")
            return []
    
    def close_position(self, position_id: str) -> Dict:
        """Close Alpaca position"""
        try:
            response = requests.delete(f"{self.base_url}/v2/positions/{position_id}", headers=self.headers)
            return {'success': response.status_code == 204}
        except Exception as e:
            logger.error(f"Error closing Alpaca position: {e}")
            return {'success': False, 'error': str(e)}


class MetaTraderBroker(BrokerBase):
    """MetaTrader API integration (simulated)"""
    
    def __init__(self, api_key: str, api_secret: str, demo_mode: bool = True):
        super().__init__(api_key, api_secret, demo_mode)
        self.server = "demo" if demo_mode else "live"
        self.account_number = api_key
    
    def connect(self) -> bool:
        """Connect to MetaTrader (simulated)"""
        try:
            # Simulate MT connection
            self.connected = True
            self.account_balance = 10000.0 if self.demo_mode else 1000.0
            logger.info(f"Connected to MetaTrader {self.server} account")
            return True
        except Exception as e:
            logger.error(f"MetaTrader connection error: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get MetaTrader account information"""
        return {
            'account_number': self.account_number,
            'balance': self.account_balance,
            'equity': self.account_balance,
            'margin': 0,
            'free_margin': self.account_balance,
            'server': self.server,
            'currency': 'USD'
        }
    
    def place_order(self, symbol: str, direction: str, amount: float, expiry_minutes: int = 1) -> Dict:
        """Place MetaTrader binary options order (simulated)"""
        try:
            order_id = f"MT_{int(time.time())}"
            
            # Simulate order placement
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'direction': direction,
                'amount': amount,
                'expiry_time': (datetime.now() + timedelta(minutes=expiry_minutes)).isoformat(),
                'status': 'placed',
                'broker': 'metatrader'
            }
            
        except Exception as e:
            logger.error(f"MetaTrader order error: {e}")
            return {
                'success': False,
                'error': str(e),
                'broker': 'metatrader'
            }
    
    def get_positions(self) -> List[Dict]:
        """Get MetaTrader positions"""
        return self.positions
    
    def close_position(self, position_id: str) -> Dict:
        """Close MetaTrader position"""
        try:
            self.positions = [p for p in self.positions if p['id'] != position_id]
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class IQOptionBroker(BrokerBase):
    """IQ Option API integration (simulated)"""
    
    def __init__(self, api_key: str, api_secret: str, demo_mode: bool = True):
        super().__init__(api_key, api_secret, demo_mode)
        self.account_type = "practice" if demo_mode else "real"
    
    def connect(self) -> bool:
        """Connect to IQ Option (simulated)"""
        try:
            self.connected = True
            self.account_balance = 10000.0 if self.demo_mode else 1000.0
            logger.info(f"Connected to IQ Option {self.account_type} account")
            return True
        except Exception as e:
            logger.error(f"IQ Option connection error: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get IQ Option account information"""
        return {
            'balance': self.account_balance,
            'account_type': self.account_type,
            'currency': 'USD',
            'country': 'US'
        }
    
    def place_order(self, symbol: str, direction: str, amount: float, expiry_minutes: int = 1) -> Dict:
        """Place IQ Option binary options order"""
        try:
            order_id = f"IQ_{int(time.time())}"
            
            # Simulate binary options order
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'direction': direction,
                'amount': amount,
                'expiry_time': (datetime.now() + timedelta(minutes=expiry_minutes)).isoformat(),
                'status': 'placed',
                'broker': 'iqoption',
                'option_type': 'binary'
            }
            
        except Exception as e:
            logger.error(f"IQ Option order error: {e}")
            return {
                'success': False,
                'error': str(e),
                'broker': 'iqoption'
            }
    
    def get_positions(self) -> List[Dict]:
        """Get IQ Option positions"""
        return self.positions
    
    def close_position(self, position_id: str) -> Dict:
        """Close IQ Option position"""
        try:
            self.positions = [p for p in self.positions if p['id'] != position_id]
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class BrokerManager:
    """Manages multiple broker connections and automated trading"""
    
    def __init__(self, db_path='signals.db'):
        self.db_path = db_path
        self.brokers = {}
        self.active_broker = None
        self.auto_trading_enabled = False
        self.risk_settings = {
            'max_trade_amount': 100.0,
            'max_daily_trades': 10,
            'min_confidence': 75,
            'allowed_tiers': ['Gold', 'Silver']
        }
        self._init_broker_tables()
    
    def _init_broker_tables(self):
        """Initialize broker-related database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS broker_accounts (
                    id INTEGER PRIMARY KEY,
                    broker_name TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    api_secret TEXT NOT NULL,
                    demo_mode BOOLEAN DEFAULT 1,
                    active BOOLEAN DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                );
                
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    broker_name TEXT,
                    symbol TEXT,
                    direction TEXT,
                    amount REAL,
                    confidence REAL,
                    tier TEXT,
                    order_id TEXT,
                    status TEXT,
                    entry_time TEXT,
                    expiry_time TEXT,
                    outcome TEXT,
                    pnl REAL,
                    created_at TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals (id)
                );
                
                CREATE TABLE IF NOT EXISTS auto_trading_settings (
                    id INTEGER PRIMARY KEY,
                    enabled BOOLEAN DEFAULT 0,
                    max_trade_amount REAL DEFAULT 100.0,
                    max_daily_trades INTEGER DEFAULT 10,
                    min_confidence REAL DEFAULT 75,
                    allowed_tiers TEXT DEFAULT '["Gold", "Silver"]',
                    risk_per_trade REAL DEFAULT 2.0,
                    updated_at TEXT
                );
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing broker tables: {e}")
    
    def add_broker(self, broker_name: str, api_key: str, api_secret: str, demo_mode: bool = True) -> bool:
        """Add a new broker connection"""
        try:
            # Create broker instance
            if broker_name.lower() == 'alpaca':
                broker = AlpacaBroker(api_key, api_secret, demo_mode)
            elif broker_name.lower() == 'metatrader':
                broker = MetaTraderBroker(api_key, api_secret, demo_mode)
            elif broker_name.lower() == 'iqoption':
                broker = IQOptionBroker(api_key, api_secret, demo_mode)
            else:
                logger.error(f"Unsupported broker: {broker_name}")
                return False
            
            # Test connection
            if broker.connect():
                self.brokers[broker_name] = broker
                
                # Save to database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO broker_accounts 
                    (broker_name, api_key, api_secret, demo_mode, active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    broker_name, api_key, api_secret, demo_mode, False,
                    datetime.now().isoformat(), datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Added broker: {broker_name}")
                return True
            else:
                logger.error(f"Failed to connect to broker: {broker_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding broker {broker_name}: {e}")
            return False
    
    def set_active_broker(self, broker_name: str) -> bool:
        """Set the active broker for trading"""
        if broker_name in self.brokers:
            self.active_broker = broker_name
            
            # Update database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Deactivate all brokers
                cursor.execute("UPDATE broker_accounts SET active = 0")
                
                # Activate selected broker
                cursor.execute(
                    "UPDATE broker_accounts SET active = 1 WHERE broker_name = ?",
                    (broker_name,)
                )
                
                conn.commit()
                conn.close()
                
                logger.info(f"Set active broker: {broker_name}")
                return True
            except Exception as e:
                logger.error(f"Error setting active broker: {e}")
                return False
        else:
            logger.error(f"Broker not found: {broker_name}")
            return False
    
    def execute_signal(self, signal_data: Dict) -> Dict:
        """Execute a signal through the active broker"""
        if not self.active_broker or self.active_broker not in self.brokers:
            return {'success': False, 'error': 'No active broker'}
        
        if not self.auto_trading_enabled:
            return {'success': False, 'error': 'Auto trading disabled'}
        
        # Check risk management rules
        if not self._check_risk_rules(signal_data):
            return {'success': False, 'error': 'Signal rejected by risk management'}
        
        try:
            broker = self.brokers[self.active_broker]
            
            # Calculate trade amount based on risk settings
            trade_amount = min(
                self.risk_settings['max_trade_amount'],
                broker.account_balance * (self.risk_settings.get('risk_per_trade', 2.0) / 100)
            )
            
            # Place order
            result = broker.place_order(
                symbol=signal_data['asset'],
                direction=signal_data['direction'],
                amount=trade_amount,
                expiry_minutes=1 if signal_data.get('expire') == '1m' else 5
            )
            
            if result['success']:
                # Record trade in database
                trade_id = f"trade_{int(time.time())}"
                self._record_trade(trade_id, signal_data, result, trade_amount)
                
                result['trade_id'] = trade_id
                result['amount'] = trade_amount
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_risk_rules(self, signal_data: Dict) -> bool:
        """Check if signal meets risk management criteria"""
        # Check confidence threshold
        if signal_data['confidence'] < self.risk_settings['min_confidence']:
            return False
        
        # Check tier allowlist
        if signal_data['tier'] not in self.risk_settings['allowed_tiers']:
            return False
        
        # Check daily trade limit
        today = datetime.now().date().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE DATE(created_at) = ? AND status = 'placed'
            """, (today,))
            
            daily_trades = cursor.fetchone()[0]
            conn.close()
            
            if daily_trades >= self.risk_settings['max_daily_trades']:
                return False
                
        except Exception as e:
            logger.error(f"Error checking daily trades: {e}")
            return False
        
        return True
    
    def _record_trade(self, trade_id: str, signal_data: Dict, order_result: Dict, amount: float):
        """Record trade in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades 
                (id, signal_id, broker_name, symbol, direction, amount, confidence, tier, 
                 order_id, status, entry_time, expiry_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                signal_data['id'],
                self.active_broker,
                signal_data['asset'],
                signal_data['direction'],
                amount,
                signal_data['confidence'],
                signal_data['tier'],
                order_result.get('order_id'),
                'placed',
                datetime.now().isoformat(),
                signal_data.get('expiry_time'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def get_account_summary(self) -> Dict:
        """Get summary of all broker accounts"""
        summary = {}
        
        for broker_name, broker in self.brokers.items():
            try:
                account_info = broker.get_account_info()
                summary[broker_name] = {
                    'connected': broker.connected,
                    'balance': account_info.get('balance', 0),
                    'demo_mode': broker.demo_mode,
                    'active': broker_name == self.active_broker
                }
            except Exception as e:
                summary[broker_name] = {
                    'connected': False,
                    'error': str(e)
                }
        
        return summary
    
    def get_trading_history(self, limit: int = 50) -> List[Dict]:
        """Get trading history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM trades 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'id': row[0],
                    'signal_id': row[1],
                    'broker_name': row[2],
                    'symbol': row[3],
                    'direction': row[4],
                    'amount': row[5],
                    'confidence': row[6],
                    'tier': row[7],
                    'order_id': row[8],
                    'status': row[9],
                    'entry_time': row[10],
                    'expiry_time': row[11],
                    'outcome': row[12],
                    'pnl': row[13],
                    'created_at': row[14]
                })
            
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trading history: {e}")
            return []
    
    def update_risk_settings(self, settings: Dict):
        """Update risk management settings"""
        self.risk_settings.update(settings)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO auto_trading_settings 
                (id, enabled, max_trade_amount, max_daily_trades, min_confidence, 
                 allowed_tiers, risk_per_trade, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.auto_trading_enabled,
                self.risk_settings['max_trade_amount'],
                self.risk_settings['max_daily_trades'],
                self.risk_settings['min_confidence'],
                json.dumps(self.risk_settings['allowed_tiers']),
                self.risk_settings.get('risk_per_trade', 2.0),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating risk settings: {e}")


# Global broker manager instance
broker_manager = BrokerManager()

def get_broker_manager() -> BrokerManager:
    """Get the global broker manager instance"""
    return broker_manager

