#!/usr/bin/env python3
"""
Reinforcement Learning Module for Premium AI Signals
Implements Q-learning and adaptive signal generation based on market outcomes.
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SignalQLearning:
    """Q-Learning implementation for signal optimization"""
    
    def __init__(self, db_path='signals.db', learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.db_path = db_path
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state -> action -> value
        self.q_table = {}
        
        # Market states and actions
        self.market_states = [
            'trending_up', 'trending_down', 'ranging', 'volatile', 
            'breakout_up', 'breakout_down', 'reversal_up', 'reversal_down'
        ]
        
        self.signal_actions = [
            'generate_gold', 'generate_silver', 'generate_bronze', 'skip_signal'
        ]
        
        # Initialize Q-table
        self._initialize_q_table()
        
        # Load historical performance data
        self._load_historical_data()
    
    def _initialize_q_table(self):
        """Initialize Q-table with default values"""
        for state in self.market_states:
            self.q_table[state] = {}
            for action in self.signal_actions:
                self.q_table[state][action] = 0.0
    
    def _load_historical_data(self):
        """Load historical signal performance for training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get signals with outcomes for training
            cursor.execute("""
                SELECT tier, confidence, outcome, pnl, timestamp 
                FROM signals 
                WHERE outcome IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 1000
            """)
            
            historical_data = cursor.fetchall()
            conn.close()
            
            # Train on historical data
            for tier, confidence, outcome, pnl, timestamp in historical_data:
                market_state = self._determine_market_state(confidence, tier)
                action = f"generate_{tier.lower()}"
                reward = self._calculate_reward(outcome, pnl, confidence)
                
                # Update Q-table
                self._update_q_value(market_state, action, reward)
                
            logger.info(f"Loaded {len(historical_data)} historical signals for RL training")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def _determine_market_state(self, confidence: float, tier: str) -> str:
        """Determine current market state based on signal characteristics"""
        # Simplified market state determination
        # In production, this would use real market data
        
        if confidence >= 85:
            return 'trending_up' if tier == 'Gold' else 'ranging'
        elif confidence >= 70:
            return 'ranging' if tier == 'Silver' else 'volatile'
        else:
            return 'volatile'
    
    def _calculate_reward(self, outcome: str, pnl: float, confidence: float) -> float:
        """Calculate reward based on signal outcome"""
        if outcome == 'win':
            # Higher reward for higher confidence wins
            base_reward = 1.0
            confidence_bonus = (confidence - 50) / 50  # 0 to 1 bonus
            pnl_bonus = min(pnl / 10, 0.5) if pnl else 0  # Cap PnL bonus
            return base_reward + confidence_bonus + pnl_bonus
        
        elif outcome == 'loss':
            # Penalty for losses, higher penalty for high confidence losses
            base_penalty = -1.0
            confidence_penalty = -(confidence - 50) / 100  # Higher confidence = higher penalty
            pnl_penalty = max(pnl / 10, -0.5) if pnl else 0  # Cap PnL penalty
            return base_penalty + confidence_penalty + pnl_penalty
        
        else:
            return 0.0  # Neutral for pending signals
    
    def _update_q_value(self, state: str, action: str, reward: float):
        """Update Q-value using Q-learning formula"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.signal_actions}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state][action]
        max_future_q = max(self.q_table[state].values()) if self.q_table[state] else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action] = new_q
    
    def get_optimal_action(self, market_state: str) -> str:
        """Get optimal action using epsilon-greedy policy"""
        if market_state not in self.q_table:
            return 'generate_silver'  # Default action
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(self.signal_actions)
        
        # Exploit: choose action with highest Q-value
        return max(self.q_table[market_state], key=self.q_table[market_state].get)
    
    def update_from_signal_outcome(self, signal_data: Dict, outcome: str, pnl: float):
        """Update Q-learning from new signal outcome"""
        market_state = self._determine_market_state(signal_data['confidence'], signal_data['tier'])
        action = f"generate_{signal_data['tier'].lower()}"
        reward = self._calculate_reward(outcome, pnl, signal_data['confidence'])
        
        self._update_q_value(market_state, action, reward)
        
        # Store updated Q-table
        self._save_q_table()
    
    def _save_q_table(self):
        """Save Q-table to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table for Q-table storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS q_learning_data (
                    id INTEGER PRIMARY KEY,
                    q_table TEXT,
                    updated_at TEXT
                )
            """)
            
            # Save Q-table as JSON
            q_table_json = json.dumps(self.q_table)
            cursor.execute("""
                INSERT OR REPLACE INTO q_learning_data (id, q_table, updated_at)
                VALUES (1, ?, ?)
            """, (q_table_json, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving Q-table: {e}")
    
    def load_q_table(self):
        """Load Q-table from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT q_table FROM q_learning_data WHERE id = 1")
            result = cursor.fetchone()
            
            if result:
                self.q_table = json.loads(result[0])
                logger.info("Loaded Q-table from database")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading Q-table: {e}")


class MarketRegimeDetector:
    """Detect market regimes for better signal adaptation"""
    
    def __init__(self):
        self.regimes = ['trending', 'ranging', 'volatile', 'breakout']
        self.current_regime = 'ranging'
        self.regime_confidence = 0.5
    
    def detect_regime(self, price_data: List[float], volume_data: List[float] = None) -> Tuple[str, float]:
        """Detect current market regime"""
        if len(price_data) < 20:
            return self.current_regime, self.regime_confidence
        
        # Calculate technical indicators for regime detection
        returns = np.diff(price_data) / price_data[:-1]
        volatility = np.std(returns)
        trend_strength = abs(np.mean(returns))
        
        # Simple regime classification
        if trend_strength > 0.001 and volatility < 0.01:
            regime = 'trending'
            confidence = min(trend_strength * 1000, 0.9)
        elif volatility > 0.02:
            regime = 'volatile'
            confidence = min(volatility * 50, 0.9)
        elif trend_strength < 0.0005 and volatility < 0.01:
            regime = 'ranging'
            confidence = 0.7
        else:
            regime = 'breakout'
            confidence = 0.6
        
        self.current_regime = regime
        self.regime_confidence = confidence
        
        return regime, confidence
    
    def get_regime_multiplier(self, signal_tier: str) -> float:
        """Get confidence multiplier based on current regime"""
        multipliers = {
            'trending': {'Gold': 1.1, 'Silver': 1.05, 'Bronze': 0.9},
            'ranging': {'Gold': 0.95, 'Silver': 1.1, 'Bronze': 1.0},
            'volatile': {'Gold': 0.8, 'Silver': 0.9, 'Bronze': 1.1},
            'breakout': {'Gold': 1.2, 'Silver': 1.0, 'Bronze': 0.8}
        }
        
        return multipliers.get(self.current_regime, {}).get(signal_tier, 1.0)


class AdaptiveSignalGenerator:
    """Adaptive signal generator using reinforcement learning"""
    
    def __init__(self, db_path='signals.db'):
        self.q_learning = SignalQLearning(db_path)
        self.regime_detector = MarketRegimeDetector()
        self.db_path = db_path
        
        # Load existing Q-table
        self.q_learning.load_q_table()
    
    def generate_adaptive_signal(self, asset: str, base_signal: Dict) -> Dict:
        """Generate signal with RL-based adaptations"""
        
        # Detect current market regime
        price_data = self._get_recent_prices(asset)
        regime, regime_confidence = self.regime_detector.detect_regime(price_data)
        
        # Determine market state for Q-learning
        market_state = self._get_market_state(base_signal, regime)
        
        # Get optimal action from Q-learning
        optimal_action = self.q_learning.get_optimal_action(market_state)
        
        # Adapt signal based on RL recommendation
        adapted_signal = self._adapt_signal(base_signal, optimal_action, regime)
        
        # Add RL metadata
        adapted_signal.update({
            'rl_action': optimal_action,
            'market_regime': regime,
            'regime_confidence': regime_confidence,
            'rl_adapted': True
        })
        
        return adapted_signal
    
    def _get_recent_prices(self, asset: str) -> List[float]:
        """Get recent price data for regime detection"""
        # Simulate price data - in production, get from real market data API
        np.random.seed(hash(asset) % 2**32)
        base_price = 1.0
        prices = [base_price]
        
        for i in range(50):
            change = np.random.normal(0, 0.001)
            prices.append(prices[-1] * (1 + change))
        
        return prices
    
    def _get_market_state(self, signal: Dict, regime: str) -> str:
        """Determine market state for Q-learning"""
        confidence = signal.get('confidence', 50)
        direction = signal.get('direction', 'CALL')
        
        if regime == 'trending':
            return 'trending_up' if direction == 'CALL' else 'trending_down'
        elif regime == 'volatile':
            return 'volatile'
        elif regime == 'breakout':
            return 'breakout_up' if direction == 'CALL' else 'breakout_down'
        else:
            return 'ranging'
    
    def _adapt_signal(self, base_signal: Dict, rl_action: str, regime: str) -> Dict:
        """Adapt signal based on RL action and market regime"""
        adapted_signal = base_signal.copy()
        
        # Apply RL action
        if rl_action == 'generate_gold':
            adapted_signal['tier'] = 'Gold'
            adapted_signal['confidence'] = min(adapted_signal['confidence'] * 1.1, 95)
        elif rl_action == 'generate_silver':
            adapted_signal['tier'] = 'Silver'
            adapted_signal['confidence'] = max(min(adapted_signal['confidence'], 85), 68)
        elif rl_action == 'generate_bronze':
            adapted_signal['tier'] = 'Bronze'
            adapted_signal['confidence'] = min(adapted_signal['confidence'], 75)
        elif rl_action == 'skip_signal':
            adapted_signal['skip'] = True
            return adapted_signal
        
        # Apply regime-based adjustments
        regime_multiplier = self.regime_detector.get_regime_multiplier(adapted_signal['tier'])
        adapted_signal['confidence'] = min(adapted_signal['confidence'] * regime_multiplier, 95)
        
        # Ensure tier consistency with confidence
        if adapted_signal['confidence'] >= 82:
            adapted_signal['tier'] = 'Gold'
        elif adapted_signal['confidence'] >= 68:
            adapted_signal['tier'] = 'Silver'
        else:
            adapted_signal['tier'] = 'Bronze'
        
        return adapted_signal
    
    def update_from_outcome(self, signal_id: str, outcome: str, pnl: float):
        """Update RL model from signal outcome"""
        try:
            # Get signal data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT asset, direction, confidence, tier, timestamp
                FROM signals WHERE id = ?
            """, (signal_id,))
            
            result = cursor.fetchone()
            if result:
                signal_data = {
                    'asset': result[0],
                    'direction': result[1],
                    'confidence': result[2],
                    'tier': result[3],
                    'timestamp': result[4]
                }
                
                # Update Q-learning
                self.q_learning.update_from_signal_outcome(signal_data, outcome, pnl)
                
                # Update signal outcome in database
                cursor.execute("""
                    UPDATE signals SET outcome = ?, pnl = ? WHERE id = ?
                """, (outcome, pnl, signal_id))
                
                conn.commit()
                logger.info(f"Updated RL model from signal {signal_id} outcome: {outcome}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating RL from outcome: {e}")


# Global instance for use in main app
adaptive_generator = AdaptiveSignalGenerator()

def get_adaptive_signal(asset: str, base_signal: Dict) -> Dict:
    """Get adaptive signal using reinforcement learning"""
    return adaptive_generator.generate_adaptive_signal(asset, base_signal)

def update_signal_outcome(signal_id: str, outcome: str, pnl: float):
    """Update signal outcome for RL training"""
    adaptive_generator.update_from_outcome(signal_id, outcome, pnl)

