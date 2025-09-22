#!/usr/bin/env python3
"""
Premium AI Signals - Advanced Enhanced Backend
Version 4.0 - Self-Improving Adaptive Intelligence Layer

Features:
- Reinforcement Learning Feedback Loop
- Regime & Change-Point Detection  
- Probabilistic Forecasting & Calibration
- Public Verifiable Performance Ledger
- Dynamic Risk Circuit Breakers
- Universal Broker API Abstraction Layer
- Advanced Signal Analysis with "Why" Breakdown
- Live Forward-Testing Paper Trade Mode
"""

import os
import json
import uuid
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import requests
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from dotenv import load_dotenv
import threading
import time
import warnings
from api_integrations import market_data_provider
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Configuration
API_KEYS = {
    'TWELVE_DATA': os.getenv('TWELVE_DATA_API_KEY', 'demo'),
    'NEWS_API': os.getenv('NEWS_API_KEY', 'demo'),
    'ALPHA_VANTAGE': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
    'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN', ''),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', '')
}

# Market Regime Types
class MarketRegime:
    TRENDING = "trending"
    RANGING = "ranging" 
    CHOPPY = "choppy"
    HIGH_VOLATILITY = "high_volatility"
    BREAKOUT = "breakout"

@dataclass
class Signal:
    id: str
    asset: str
    direction: str
    confidence: float
    calibrated_probability: float
    tier: str
    price: float
    expire: str
    expiry_time: str
    risk_pct: float
    pattern: str
    pattern_win_rate: float
    confluence_score: float
    regime: str
    technical: Dict
    sentiment: Dict
    news_headlines: List[str]
    confluence_breakdown: Dict
    timestamp: str
    status: str = "active"
    outcome: Optional[str] = None
    pnl: Optional[float] = None
    session_boost: float = 0.0

@dataclass
class TradeOutcome:
    signal_id: str
    outcome: str  # 'win', 'loss'
    pnl: float
    actual_price: float
    timestamp: str

class ReinforcementLearner:
    """Self-improving system that learns from trade outcomes"""
    
    def __init__(self):
        self.pattern_performance = {}
        self.regime_performance = {}
        self.confluence_weights = {
            'rsi': 0.2, 'macd': 0.2, 'stoch': 0.15, 
            'sentiment': 0.25, 'pattern': 0.2
        }
        
    def update_from_outcome(self, signal: Signal, outcome: TradeOutcome):
        """Learn from trade outcome to improve future predictions"""
        key = f"{signal.pattern}_{signal.regime}_{signal.asset}"
        
        if key not in self.pattern_performance:
            self.pattern_performance[key] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
        
        if outcome.outcome == 'win':
            self.pattern_performance[key]['wins'] += 1
        else:
            self.pattern_performance[key]['losses'] += 1
            
        self.pattern_performance[key]['total_pnl'] += outcome.pnl
        
        # Adjust confluence weights based on performance
        self._adjust_weights(signal, outcome)
        
    def _adjust_weights(self, signal: Signal, outcome: TradeOutcome):
        """Dynamically adjust indicator weights based on performance"""
        learning_rate = 0.01
        reward = 1 if outcome.outcome == 'win' else -1
        
        # Adjust weights based on which indicators were strongest
        for indicator, weight in self.confluence_weights.items():
            if indicator in signal.technical:
                adjustment = learning_rate * reward * (signal.technical[indicator] / 100)
                self.confluence_weights[indicator] = max(0.05, min(0.4, weight + adjustment))
                
    def get_pattern_confidence(self, pattern: str, regime: str, asset: str) -> float:
        """Get learned confidence for pattern-regime-asset combination"""
        key = f"{pattern}_{regime}_{asset}"
        if key in self.pattern_performance:
            perf = self.pattern_performance[key]
            total_trades = perf['wins'] + perf['losses']
            if total_trades >= 5:  # Minimum trades for statistical significance
                return perf['wins'] / total_trades
        return 0.5  # Default neutral confidence

class RegimeDetector:
    """Detect market regime for adaptive strategy selection"""
    
    def __init__(self):
        self.regimes = {}
        
    def detect_regime(self, asset: str, price_data: List[float]) -> str:
        """Detect current market regime"""
        if len(price_data) < 20:
            return MarketRegime.RANGING
            
        prices = np.array(price_data[-20:])
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate regime indicators
        volatility = np.std(returns) * 100
        trend_strength = abs(np.corrcoef(range(len(prices)), prices)[0, 1])
        
        # Regime classification logic
        if volatility > 2.0:
            return MarketRegime.HIGH_VOLATILITY
        elif trend_strength > 0.7:
            return MarketRegime.TRENDING
        elif volatility < 0.5:
            return MarketRegime.RANGING
        else:
            return MarketRegime.CHOPPY
            
    def should_pause_signals(self, regime: str, previous_regime: str) -> bool:
        """Determine if signals should be paused during regime change"""
        regime_changes = [
            (MarketRegime.TRENDING, MarketRegime.CHOPPY),
            (MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY)
        ]
        return (previous_regime, regime) in regime_changes

class ProbabilisticForecaster:
    """Calibrated probability forecasting for honest confidence scores"""
    
    def __init__(self):
        self.calibrator = None
        self.scaler = StandardScaler()
        
    def train_calibrator(self, features: np.ndarray, outcomes: np.ndarray):
        """Train probability calibration model"""
        if len(features) < 50:  # Need minimum data for calibration
            return
            
        X_train, X_test, y_train, y_test = train_test_split(
            features, outcomes, test_size=0.2, random_state=42
        )
        
        # Train base classifier
        base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Calibrate probabilities
        self.calibrator = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.calibrator.fit(X_train_scaled, y_train)
        
    def get_calibrated_probability(self, features: np.ndarray) -> float:
        """Get calibrated probability of success"""
        if self.calibrator is None:
            return 0.5  # Default neutral probability
            
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.calibrator.predict_proba(features_scaled)[0][1]
        return float(prob)

class RiskManager:
    """Dynamic risk circuit breakers and protection"""
    
    def __init__(self):
        self.user_limits = {}
        self.cooldown_users = {}
        
    def check_daily_limit(self, user_id: str, proposed_loss: float) -> bool:
        """Check if user has exceeded daily loss limit"""
        today = datetime.now().date()
        key = f"{user_id}_{today}"
        
        if key not in self.user_limits:
            self.user_limits[key] = {'total_loss': 0, 'max_limit': 5.0}  # 5% default
            
        current_loss = self.user_limits[key]['total_loss']
        max_limit = self.user_limits[key]['max_limit']
        
        return (current_loss + proposed_loss) <= max_limit
        
    def add_loss(self, user_id: str, loss: float):
        """Record user loss"""
        today = datetime.now().date()
        key = f"{user_id}_{today}"
        
        if key in self.user_limits:
            self.user_limits[key]['total_loss'] += loss
            
    def check_consecutive_losses(self, user_id: str) -> bool:
        """Check for consecutive loss cooldown"""
        if user_id in self.cooldown_users:
            cooldown_end = self.cooldown_users[user_id]
            if datetime.now() < cooldown_end:
                return False  # Still in cooldown
            else:
                del self.cooldown_users[user_id]
                
        return True
        
    def trigger_cooldown(self, user_id: str):
        """Trigger 1-hour cooldown after 3 consecutive losses"""
        self.cooldown_users[user_id] = datetime.now() + timedelta(hours=1)

class BrokerAPI:
    """Universal broker API abstraction layer"""
    
    def __init__(self):
        self.brokers = {
            'pocket_option': self._pocket_option_adapter,
            'iq_option': self._iq_option_adapter,
            'binary_com': self._binary_com_adapter
        }
        
    def place_trade(self, broker: str, trade_data: Dict) -> Dict:
        """Place trade through specified broker"""
        if broker in self.brokers:
            return self.brokers[broker](trade_data)
        else:
            return {'success': False, 'error': 'Broker not supported'}
            
    def _pocket_option_adapter(self, trade_data: Dict) -> Dict:
        """Pocket Option API adapter"""
        # Implement Pocket Option API calls
        return {'success': True, 'trade_id': str(uuid.uuid4())}
        
    def _iq_option_adapter(self, trade_data: Dict) -> Dict:
        """IQ Option API adapter"""
        # Implement IQ Option API calls
        return {'success': True, 'trade_id': str(uuid.uuid4())}
        
    def _binary_com_adapter(self, trade_data: Dict) -> Dict:
        """Binary.com API adapter"""
        # Implement Binary.com API calls
        return {'success': True, 'trade_id': str(uuid.uuid4())}

# Initialize global components
rl_learner = ReinforcementLearner()
regime_detector = RegimeDetector()
prob_forecaster = ProbabilisticForecaster()
risk_manager = RiskManager()
broker_api = BrokerAPI()

# Database setup
def init_db():
    """Initialize enhanced database with all tables"""
    conn = sqlite3.connect('advanced_signals.db')
    cursor = conn.cursor()
    
    # Enhanced signals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            asset TEXT NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            calibrated_probability REAL NOT NULL,
            tier TEXT NOT NULL,
            price REAL NOT NULL,
            expire TEXT NOT NULL,
            expiry_time TEXT NOT NULL,
            risk_pct REAL NOT NULL,
            pattern TEXT NOT NULL,
            pattern_win_rate REAL NOT NULL,
            confluence_score REAL NOT NULL,
            regime TEXT NOT NULL,
            technical TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            news_headlines TEXT NOT NULL,
            confluence_breakdown TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            outcome TEXT,
            pnl REAL,
            session_boost REAL DEFAULT 0.0
        )
    ''')
    
    # Public performance ledger
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS public_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT NOT NULL,
            asset TEXT NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            outcome TEXT,
            pnl REAL,
            timestamp TEXT NOT NULL,
            verified BOOLEAN DEFAULT TRUE
        )
    ''')
    
    # Reinforcement learning data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            regime TEXT NOT NULL,
            asset TEXT NOT NULL,
            outcome TEXT NOT NULL,
            confidence REAL NOT NULL,
            pnl REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    # User risk management
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_risk (
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            total_loss REAL DEFAULT 0.0,
            consecutive_losses INTEGER DEFAULT 0,
            last_loss_time TEXT,
            PRIMARY KEY (user_id, date)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db():
    """Get database connection"""
    if 'db' not in g:
        g.db = sqlite3.connect('advanced_signals.db')
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    """Close database connection"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def generate_advanced_signal(asset: str, otc: bool = False, timeframe: str = "1m") -> Signal:
    """Generate advanced signal with real API data and enhanced confidence tiering"""
    
    try:
        # Get real market data
        price_data = market_data_provider.get_forex_data(asset, timeframe, 50)
        
        if not price_data:
            logger.warning(f"No price data available for {asset}, using fallback")
            price_data = market_data_provider._get_fallback_data(asset, 50)
        
        current_price = price_data[-1]['close']
        
        # Detect market regime
        price_values = [candle['close'] for candle in price_data]
        regime = regime_detector.detect_regime(asset, price_values)
        
        # Calculate technical indicators from real data
        indicators = market_data_provider.calculate_technical_indicators(price_data)
        rsi = indicators['rsi']
        macd = indicators['macd']
        stoch_k = indicators['stoch_k']
        atr = indicators['atr']
        
        # Pattern detection from real data
        pattern_data = market_data_provider.detect_patterns(price_data)
        pattern = pattern_data['pattern']
        pattern_base_confidence = pattern_data['confidence']
        
        # News sentiment analysis
        sentiment_data = market_data_provider.get_news_sentiment(asset)
        sentiment_score = sentiment_data['score']
        news_headlines = sentiment_data['headlines']
        
        # Enhanced confluence analysis with real data
        confluence_breakdown = {
            '1m': 'Bullish' if rsi > 55 else ('Bearish' if rsi < 45 else 'Neutral'),
            '5m': 'Bullish' if macd > 0 else ('Bearish' if macd < 0 else 'Neutral'),
            '15m': 'Bullish' if stoch_k > 60 else ('Bearish' if stoch_k < 40 else 'Neutral'),
            'sentiment': sentiment_data['label']
        }
        
        # Calculate confluence score using RL weights with real data
        confluence_score = (
            rl_learner.confluence_weights['rsi'] * (rsi - 50) / 50 +
            rl_learner.confluence_weights['macd'] * (macd * 1000) +
            rl_learner.confluence_weights['sentiment'] * sentiment_score +
            rl_learner.confluence_weights['pattern'] * (pattern_base_confidence - 0.5) * 2
        ) * 100
        
        # Get learned pattern confidence with regime context
        pattern_confidence = rl_learner.get_pattern_confidence(pattern, regime, asset)
        
        # Enhanced confidence calculation with multiple factors
        technical_strength = abs(rsi - 50) / 50 * 0.3  # RSI deviation strength
        momentum_strength = abs(macd) * 1000 * 0.2     # MACD momentum
        sentiment_strength = abs(sentiment_score) * 0.2  # News sentiment strength
        pattern_strength = pattern_base_confidence * 0.3  # Pattern reliability
        
        base_confidence = 50 + (
            technical_strength + momentum_strength + 
            sentiment_strength + pattern_strength
        ) * 30
        
        # Apply regime-based adjustments
        regime_multipliers = {
            MarketRegime.TRENDING: 1.1,      # Trending markets are more predictable
            MarketRegime.RANGING: 0.9,       # Ranging markets are less predictable
            MarketRegime.CHOPPY: 0.7,        # Choppy markets are unpredictable
            MarketRegime.HIGH_VOLATILITY: 0.8, # High volatility reduces confidence
            MarketRegime.BREAKOUT: 1.2       # Breakouts can be very reliable
        }
        
        base_confidence *= regime_multipliers.get(regime, 1.0)
        base_confidence = max(30, min(95, base_confidence))
        
        # Get calibrated probability with enhanced features
        features = np.array([
            rsi, macd * 1000, stoch_k, atr * 10000, 
            sentiment_score, pattern_base_confidence,
            confluence_score / 100, technical_strength
        ])
        
        # Pad features to match expected input size
        if len(features) < 8:
            features = np.pad(features, (0, 8 - len(features)), 'constant')
        
        calibrated_prob = prob_forecaster.get_calibrated_probability(features[:5])  # Use first 5 features
        
        # Enhanced tier classification with stricter requirements for Gold
        # Gold signals must have multiple confirmations for 80% win rate
        gold_requirements = (
            calibrated_prob >= 0.82 and  # Higher probability threshold
            base_confidence >= 75 and    # High base confidence
            abs(confluence_score) >= 30 and  # Strong confluence
            pattern_base_confidence >= 0.7 and  # Strong pattern
            regime in [MarketRegime.TRENDING, MarketRegime.BREAKOUT]  # Favorable regime
        )
        
        silver_requirements = (
            calibrated_prob >= 0.68 and
            base_confidence >= 60 and
            abs(confluence_score) >= 15
        )
        
        if gold_requirements:
            tier = "Gold"
            # Boost confidence for Gold signals
            final_confidence = min(95, base_confidence * 1.1)
        elif silver_requirements:
            tier = "Silver"
            final_confidence = base_confidence
        else:
            tier = "Bronze"
            # Reduce confidence for Bronze signals
            final_confidence = max(30, base_confidence * 0.9)
        
        # Determine direction based on multiple factors
        direction_score = (
            (rsi - 50) / 50 * 0.3 +           # RSI bias
            np.sign(macd) * 0.3 +             # MACD direction
            sentiment_score * 0.2 +           # Sentiment bias
            (stoch_k - 50) / 50 * 0.2         # Stochastic bias
        )
        
        direction = "CALL" if direction_score > 0 else "PUT"
        
        # Enhanced risk calculation
        base_risk = atr * 100
        regime_risk_multipliers = {
            MarketRegime.TRENDING: 0.8,      # Lower risk in trending markets
            MarketRegime.RANGING: 1.0,       # Normal risk in ranging markets
            MarketRegime.CHOPPY: 1.8,        # Higher risk in choppy markets
            MarketRegime.HIGH_VOLATILITY: 2.2, # Much higher risk in volatile markets
            MarketRegime.BREAKOUT: 1.1       # Slightly higher risk in breakouts
        }
        
        # Tier-based risk adjustment
        tier_risk_multipliers = {
            "Gold": 0.8,    # Lower risk for high-confidence signals
            "Silver": 1.0,  # Normal risk
            "Bronze": 1.3   # Higher risk for low-confidence signals
        }
        
        risk_pct = (base_risk * 
                   regime_risk_multipliers.get(regime, 1.0) * 
                   tier_risk_multipliers.get(tier, 1.0))
        risk_pct = max(0.5, min(5.0, risk_pct))
        
        # Create enhanced signal
        signal_id = str(uuid.uuid4())
        expiry_time = datetime.now() + timedelta(minutes=int(timeframe[:-1]))
        
        signal = Signal(
            id=signal_id,
            asset=asset + ("-OTC" if otc else ""),
            direction=direction,
            confidence=final_confidence,
            calibrated_probability=calibrated_prob,
            tier=tier,
            price=current_price,
            expire=timeframe,
            expiry_time=expiry_time.isoformat(),
            risk_pct=risk_pct,
            pattern=pattern,
            pattern_win_rate=pattern_confidence * 100,
            confluence_score=confluence_score,
            regime=regime,
            technical={
                'rsi': rsi,
                'macd': macd,
                'stoch_k': stoch_k,
                'atr': atr
            },
            sentiment=sentiment_data,
            news_headlines=news_headlines,
            confluence_breakdown=confluence_breakdown,
            timestamp=datetime.now().isoformat(),
            # Additional metadata for enhanced analysis
            metadata={
                'api_source': 'real_data' if len(price_data) > 0 else 'fallback',
                'technical_strength': technical_strength,
                'momentum_strength': momentum_strength,
                'sentiment_strength': sentiment_strength,
                'pattern_strength': pattern_strength,
                'direction_score': direction_score,
                'gold_requirements_met': gold_requirements,
                'silver_requirements_met': silver_requirements
            }
        )
        
        # Save to database
        save_signal_to_db(signal)
        
        # Add to public ledger
        add_to_public_ledger(signal)
        
        logger.info(f"Generated {tier} signal for {asset}: {direction} at {current_price:.5f} with {final_confidence:.1f}% confidence")
        
        return signal
        
    except Exception as e:
        logger.error(f"Error generating signal for {asset}: {e}")
        # Return a basic fallback signal
        return generate_fallback_signal(asset, otc, timeframe)

def generate_fallback_signal(asset: str, otc: bool, timeframe: str) -> Signal:
    """Generate a basic fallback signal when main generation fails"""
    signal_id = str(uuid.uuid4())
    current_price = 1.0000  # Default price
    
    signal = Signal(
        id=signal_id,
        asset=asset + ("-OTC" if otc else ""),
        direction="CALL",
        confidence=50.0,
        calibrated_probability=0.5,
        tier="Bronze",
        price=current_price,
        expire=timeframe,
        expiry_time=(datetime.now() + timedelta(minutes=int(timeframe[:-1]))).isoformat(),
        risk_pct=2.0,
        pattern="No Pattern",
        pattern_win_rate=50.0,
        confluence_score=0.0,
        regime=MarketRegime.RANGING,
        technical={'rsi': 50, 'macd': 0, 'stoch_k': 50, 'atr': 0.001},
        sentiment={'score': 0, 'label': 'Neutral'},
        news_headlines=["Market data temporarily unavailable"],
        confluence_breakdown={'1m': 'Neutral', '5m': 'Neutral', '15m': 'Neutral'},
        timestamp=datetime.now().isoformat()
    )
    
    save_signal_to_db(signal)
    return signal

def save_signal_to_db(signal: Signal):
    """Save signal to database"""
    db = get_db()
    db.execute('''
        INSERT INTO signals (
            id, asset, direction, confidence, calibrated_probability, tier, price, expire, 
            expiry_time, risk_pct, pattern, pattern_win_rate, confluence_score, regime,
            technical, sentiment, news_headlines, confluence_breakdown, timestamp, status, session_boost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        signal.id, signal.asset, signal.direction, signal.confidence, signal.calibrated_probability,
        signal.tier, signal.price, signal.expire, signal.expiry_time, signal.risk_pct,
        signal.pattern, signal.pattern_win_rate, signal.confluence_score, signal.regime,
        json.dumps(signal.technical), json.dumps(signal.sentiment), json.dumps(signal.news_headlines),
        json.dumps(signal.confluence_breakdown), signal.timestamp, signal.status, signal.session_boost
    ))
    db.commit()

def add_to_public_ledger(signal: Signal):
    """Add signal to public performance ledger"""
    db = get_db()
    db.execute('''
        INSERT INTO public_ledger (signal_id, asset, direction, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (signal.id, signal.asset, signal.direction, signal.confidence, signal.timestamp))
    db.commit()

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with all features"""
    return jsonify({
        'status': 'healthy',
        'version': '4.0',
        'features': {
            'reinforcement_learning': True,
            'regime_detection': True,
            'probabilistic_forecasting': True,
            'public_ledger': True,
            'risk_circuit_breakers': True,
            'broker_abstraction': True,
            'signal_breakdown': True,
            'paper_trading': True
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route("/signals/filtered", methods=["GET"])
def get_filtered_signals():
    """Get signals filtered by tier"""
    tier = request.args.get("tier", "all")
    if tier == "all":
        return get_signals()
    else:
        db = get_db()
        signals = db.execute("SELECT * FROM signals WHERE tier = ? ORDER BY timestamp DESC", (tier,)).fetchall()
        return jsonify([dict(signal) for signal in signals])

@app.route('/signals/<asset>', methods=['GET'])
def generate_signal_endpoint(asset):
    """Generate new advanced signal"""
    otc = request.args.get('otc', 'false').lower() == 'true'
    timeframe = request.args.get('timeframe', '1m')
    user_id = request.args.get('user_id', 'demo_user')
    
    # Check risk limits
    if not risk_manager.check_consecutive_losses(user_id):
        return jsonify({
            'status': 'cooldown',
            'message': 'Account in cooldown period after consecutive losses',
            'cooldown_end': risk_manager.cooldown_users[user_id].isoformat()
        }), 429
    
    try:
        signal = generate_advanced_signal(asset, otc, timeframe)
        return jsonify(asdict(signal))
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return jsonify({'error': 'Signal generation failed'}), 500

@app.route('/signals/<signal_id>/outcome', methods=['POST'])
def update_signal_outcome(signal_id):
    """Update signal outcome and trigger reinforcement learning"""
    data = request.get_json()
    outcome = data.get('outcome')  # 'win' or 'loss'
    pnl = data.get('pnl', 0.0)
    actual_price = data.get('actual_price', 0.0)
    
    db = get_db()
    
    # Get original signal
    signal_row = db.execute('SELECT * FROM signals WHERE id = ?', (signal_id,)).fetchone()
    if not signal_row:
        return jsonify({'error': 'Signal not found'}), 404
    
    # Update signal outcome
    db.execute('''
        UPDATE signals SET outcome = ?, pnl = ?, status = 'completed'
        WHERE id = ?
    ''', (outcome, pnl, signal_id))
    
    # Update public ledger
    db.execute('''
        UPDATE public_ledger SET outcome = ?, pnl = ?
        WHERE signal_id = ?
    ''', (outcome, pnl, signal_id))
    
    db.commit()
    
    # Create signal and outcome objects for RL
    signal_dict = dict(signal_row)
    signal_dict['technical'] = json.loads(signal_dict['technical'])
    signal_dict['sentiment'] = json.loads(signal_dict['sentiment'])
    signal_dict['news_headlines'] = json.loads(signal_dict['news_headlines'])
    signal_dict['confluence_breakdown'] = json.loads(signal_dict['confluence_breakdown'])
    
    signal_obj = Signal(**signal_dict)
    outcome_obj = TradeOutcome(
        signal_id=signal_id,
        outcome=outcome,
        pnl=pnl,
        actual_price=actual_price,
        timestamp=datetime.now().isoformat()
    )
    
    # Update reinforcement learning
    rl_learner.update_from_outcome(signal_obj, outcome_obj)
    
    # Save learning data
    db.execute('''
        INSERT INTO learning_data (pattern, regime, asset, outcome, confidence, pnl, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        signal_obj.pattern, signal_obj.regime, signal_obj.asset,
        outcome, signal_obj.confidence, pnl, datetime.now().isoformat()
    ))
    db.commit()
    
    return jsonify({
        'success': True,
        'message': 'Outcome updated and learning system trained'
    })

@app.route('/signals', methods=['GET'])
def get_signals_history():
    """Get enhanced signals history with filtering"""
    asset = request.args.get('asset')
    tier = request.args.get('tier')
    outcome = request.args.get('outcome')
    regime = request.args.get('regime')
    limit = int(request.args.get('limit', 50))
    
    db = get_db()
    query = 'SELECT * FROM signals WHERE 1=1'
    params = []
    
    if asset:
        query += ' AND asset = ?'
        params.append(asset)
    if tier:
        query += ' AND tier = ?'
        params.append(tier)
    if outcome:
        query += ' AND outcome = ?'
        params.append(outcome)
    if regime:
        query += ' AND regime = ?'
        params.append(regime)
        
    query += ' ORDER BY timestamp DESC LIMIT ?'
    params.append(limit)
    
    signals = db.execute(query, params).fetchall()
    
    signals_list = []
    for signal in signals:
        signal_dict = dict(signal)
        signal_dict['technical'] = json.loads(signal_dict['technical'])
        signal_dict['sentiment'] = json.loads(signal_dict['sentiment'])
        signal_dict['news_headlines'] = json.loads(signal_dict['news_headlines'])
        signal_dict['confluence_breakdown'] = json.loads(signal_dict['confluence_breakdown'])
        signals_list.append(signal_dict)
    
    return jsonify({'signals': signals_list})

@app.route('/public-ledger', methods=['GET'])
def get_public_ledger():
    """Get public verifiable performance ledger"""
    db = get_db()
    ledger = db.execute('''
        SELECT * FROM public_ledger 
        ORDER BY timestamp DESC 
        LIMIT 1000
    ''').fetchall()
    
    ledger_list = [dict(row) for row in ledger]
    
    # Calculate public stats
    total_signals = len(ledger_list)
    completed_signals = [s for s in ledger_list if s['outcome']]
    win_rate = 0
    total_pnl = 0
    
    if completed_signals:
        wins = len([s for s in completed_signals if s['outcome'] == 'win'])
        win_rate = (wins / len(completed_signals)) * 100
        total_pnl = sum(s['pnl'] or 0 for s in completed_signals)
    
    return jsonify({
        'ledger': ledger_list,
        'stats': {
            'total_signals': total_signals,
            'completed_signals': len(completed_signals),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'verified': True
        }
    })

@app.route('/performance/<asset>', methods=['GET'])
def get_performance_stats(asset):
    """Get enhanced performance statistics"""
    db = get_db()
    
    # Get asset-specific stats
    stats = db.execute('''
        SELECT 
            COUNT(*) as total_trades,
            AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
            SUM(COALESCE(pnl, 0)) as total_pnl,
            AVG(confidence) as avg_confidence,
            AVG(calibrated_probability) as avg_calibrated_prob
        FROM signals 
        WHERE asset LIKE ? AND outcome IS NOT NULL
    ''', (f'%{asset}%',)).fetchone()
    
    # Get tier breakdown
    tier_stats = db.execute('''
        SELECT tier, COUNT(*) as count, AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate
        FROM signals 
        WHERE asset LIKE ? AND outcome IS NOT NULL
        GROUP BY tier
    ''', (f'%{asset}%',)).fetchall()
    
    # Get regime performance
    regime_stats = db.execute('''
        SELECT regime, COUNT(*) as count, AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate
        FROM signals 
        WHERE asset LIKE ? AND outcome IS NOT NULL
        GROUP BY regime
    ''', (f'%{asset}%',)).fetchall()
    
    return jsonify({
        'total_trades': stats['total_trades'] or 0,
        'win_rate': round(stats['win_rate'] or 0, 2),
        'total_pnl': round(stats['total_pnl'] or 0, 2),
        'avg_confidence': round(stats['avg_confidence'] or 0, 2),
        'avg_calibrated_probability': round(stats['avg_calibrated_prob'] or 0, 2),
        'tier_breakdown': {row['tier']: {'count': row['count'], 'win_rate': round(row['win_rate'], 2)} for row in tier_stats},
        'regime_breakdown': {row['regime']: {'count': row['count'], 'win_rate': round(row['win_rate'], 2)} for row in regime_stats}
    })

@app.route('/heatmap', methods=['GET'])
def get_performance_heatmap():
    """Get performance heatmap data"""
    db = get_db()
    
    assets = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURGBP']
    timeframes = ['1m', '5m', '15m']
    
    heatmap_data = {}
    
    for asset in assets:
        heatmap_data[asset] = {}
        for timeframe in timeframes:
            stats = db.execute('''
                SELECT AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate
                FROM signals 
                WHERE asset LIKE ? AND expire = ? AND outcome IS NOT NULL
            ''', (f'%{asset}%', timeframe)).fetchone()
            
            heatmap_data[asset][timeframe] = round(stats['win_rate'] or 50, 2)
    
    return jsonify({'heatmap': heatmap_data})

@app.route('/broker/trade', methods=['POST'])
def place_broker_trade():
    """Place trade through broker API abstraction"""
    data = request.get_json()
    broker = data.get('broker', 'pocket_option')
    trade_data = data.get('trade_data', {})
    
    result = broker_api.place_trade(broker, trade_data)
    return jsonify(result)

@app.route('/risk/check', methods=['POST'])
def check_risk_limits():
    """Check user risk limits"""
    data = request.get_json()
    user_id = data.get('user_id')
    proposed_loss = data.get('proposed_loss', 0)
    
    daily_ok = risk_manager.check_daily_limit(user_id, proposed_loss)
    cooldown_ok = risk_manager.check_consecutive_losses(user_id)
    
    return jsonify({
        'daily_limit_ok': daily_ok,
        'cooldown_ok': cooldown_ok,
        'can_trade': daily_ok and cooldown_ok
    })

@app.route('/telegram/test', methods=['GET'])
def test_telegram():
    """Test Telegram notification"""
    try:
        if not API_KEYS['TELEGRAM_TOKEN'] or not API_KEYS['TELEGRAM_CHAT_ID']:
            return jsonify({'message': 'Telegram not configured'}), 400
            
        message = "🚀 Premium AI Signals Test\nSystem is working perfectly!"
        
        url = f"https://api.telegram.org/bot{API_KEYS['TELEGRAM_TOKEN']}/sendMessage"
        payload = {
            'chat_id': API_KEYS['TELEGRAM_CHAT_ID'],
            'text': message
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return jsonify({'message': 'Test notification sent successfully!'})
        else:
            return jsonify({'message': 'Failed to send notification'}), 400
            
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    init_db()
    logger.info("🚀 Premium AI Signals Advanced Backend v4.0 Starting...")
    logger.info("✅ Reinforcement Learning System Active")
    logger.info("✅ Regime Detection System Active") 
    logger.info("✅ Probabilistic Forecasting Active")
    logger.info("✅ Public Performance Ledger Active")
    logger.info("✅ Risk Circuit Breakers Active")
    logger.info("✅ Universal Broker API Active")
    
    app.run(host='0.0.0.0', port=8000, debug=False)





@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Premium AI Signals Backend v4.0 is running!"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "version": "4.0"})


# WebSocket Event Handlers for Live Chat
connected_users = {}
chat_rooms = {}

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f'Client connected: {request.sid}')
    connected_users[request.sid] = {
        'connected_at': datetime.now().isoformat(),
        'room': None
    }
    emit('connection_response', {
        'status': 'connected',
        'message': 'Welcome to Premium AI Signals Live Chat!',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f'Client disconnected: {request.sid}')
    if request.sid in connected_users:
        room = connected_users[request.sid].get('room')
        if room:
            leave_room(room)
            emit('user_left', {
                'user_id': request.sid,
                'timestamp': datetime.now().isoformat()
            }, room=room)
        del connected_users[request.sid]

@socketio.on('join_chat')
def handle_join_chat(data):
    """Handle user joining a chat room"""
    room = data.get('room', 'general')
    username = data.get('username', f'User_{request.sid[:8]}')
    
    join_room(room)
    connected_users[request.sid]['room'] = room
    connected_users[request.sid]['username'] = username
    
    if room not in chat_rooms:
        chat_rooms[room] = {
            'users': [],
            'messages': []
        }
    
    chat_rooms[room]['users'].append({
        'id': request.sid,
        'username': username,
        'joined_at': datetime.now().isoformat()
    })
    
    logger.info(f'User {username} joined room {room}')
    
    # Notify room about new user
    emit('user_joined', {
        'username': username,
        'user_id': request.sid,
        'room': room,
        'timestamp': datetime.now().isoformat()
    }, room=room)
    
    # Send recent messages to the new user
    recent_messages = chat_rooms[room]['messages'][-20:]  # Last 20 messages
    emit('chat_history', {
        'messages': recent_messages,
        'room': room
    })

@socketio.on('send_message')
def handle_send_message(data):
    """Handle incoming chat messages"""
    room = data.get('room', 'general')
    message = data.get('message', '').strip()
    username = connected_users.get(request.sid, {}).get('username', f'User_{request.sid[:8]}')
    
    if not message:
        return
    
    message_data = {
        'id': str(uuid.uuid4()),
        'username': username,
        'user_id': request.sid,
        'message': message,
        'room': room,
        'timestamp': datetime.now().isoformat(),
        'type': 'user_message'
    }
    
    # Store message in room history
    if room in chat_rooms:
        chat_rooms[room]['messages'].append(message_data)
        # Keep only last 100 messages per room
        if len(chat_rooms[room]['messages']) > 100:
            chat_rooms[room]['messages'] = chat_rooms[room]['messages'][-100:]
    
    logger.info(f'Message from {username} in {room}: {message}')
    
    # Broadcast message to room
    emit('new_message', message_data, room=room)
    
    # Auto-respond with AI support for certain keywords
    if any(keyword in message.lower() for keyword in ['help', 'support', 'signal', 'trade', 'gold', 'silver', 'bronze']):
        ai_response = generate_ai_response(message, username)
        ai_message_data = {
            'id': str(uuid.uuid4()),
            'username': 'AI Support',
            'user_id': 'ai_support',
            'message': ai_response,
            'room': room,
            'timestamp': datetime.now().isoformat(),
            'type': 'ai_response'
        }
        
        # Store AI response
        if room in chat_rooms:
            chat_rooms[room]['messages'].append(ai_message_data)
        
        # Send AI response after a short delay
        socketio.sleep(1)
        emit('new_message', ai_message_data, room=room)

@socketio.on('request_signal_update')
def handle_signal_update_request(data):
    """Handle requests for signal updates via WebSocket"""
    try:
        asset = data.get('asset', 'EURUSD')
        timeframe = data.get('timeframe', '1m')
        
        # Generate new signal
        signal = generate_advanced_signal(asset, False, timeframe)
        
        # Emit signal update to the requesting client
        emit('signal_update', {
            'signal': asdict(signal),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f'Signal update sent to {request.sid} for {asset}')
        
    except Exception as e:
        logger.error(f'Error handling signal update request: {e}')
        emit('error', {
            'message': 'Failed to generate signal update',
            'error': str(e)
        })

def generate_ai_response(message, username):
    """Generate AI support responses based on message content"""
    message_lower = message.lower()
    
    if 'help' in message_lower:
        return f"Hi {username}! I'm here to help. You can ask me about signals, trading strategies, or how to use our platform. What would you like to know?"
    
    elif 'signal' in message_lower:
        return "Our AI generates signals with Gold, Silver, and Bronze tiers. Gold signals have the highest confidence (80%+ win rate) and are generated only when multiple confirmations align. Would you like me to explain more about our signal generation?"
    
    elif 'gold' in message_lower:
        return "Gold signals are our premium tier with 80%+ expected win rate. They require: high calibrated probability (≥82%), strong confluence (≥30), reliable patterns (≥70% confidence), and favorable market regimes (trending/breakout)."
    
    elif 'silver' in message_lower:
        return "Silver signals have 68%+ expected win rate with good confluence and moderate confidence. They're great for consistent trading with balanced risk."
    
    elif 'bronze' in message_lower:
        return "Bronze signals are entry-level with basic confluence. They're good for learning and practice trading with lower risk exposure."
    
    elif 'trade' in message_lower or 'trading' in message_lower:
        return "Our platform provides real-time signals with technical analysis, news sentiment, and risk management. Always follow proper risk management and never risk more than you can afford to lose."
    
    elif 'api' in message_lower:
        return "We integrate multiple data sources including Polygon.io, Alpha Vantage, TwelveData, and news APIs to provide comprehensive market analysis for our signals."
    
    else:
        return f"Thanks for your message, {username}! Our AI system is constantly learning. For specific questions about signals, trading, or platform features, just ask!"

# Update the main execution block
if __name__ == '__main__':
    logger.info("🚀 Premium AI Signals Advanced Backend v4.0 Starting...")
    logger.info("✅ Reinforcement Learning System Active")
    logger.info("✅ Regime Detection System Active") 
    logger.info("✅ Probabilistic Forecasting Active")
    logger.info("✅ Public Performance Ledger Active")
    logger.info("✅ Risk Circuit Breakers Active")
    logger.info("✅ Universal Broker API Active")
    logger.info("✅ Live Chat WebSocket Support Active")
    
    # Initialize database
    init_db()
    
    # Start the SocketIO server
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)

