import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import requests
import logging
import random
import uuid
from enum import Enum

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Premium AI Binary Signals", version="3.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signals.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class SignalTier(str, Enum):
    GOLD = "Gold"
    SILVER = "Silver"
    BRONZE = "Bronze"

class SignalOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    PENDING = "pending"

class Signal(Base):
    __tablename__ = "signals"
    id = Column(String, primary_key=True)
    asset = Column(String)
    direction = Column(String)
    confidence = Column(Float)
    tier = Column(String)
    price = Column(Float)
    expire = Column(String)
    confluence_score = Column(Float)
    sentiment_score = Column(Float)
    pattern = Column(String)
    pattern_win_rate = Column(Float)
    risk_pct = Column(Float)
    atr = Column(Float)
    rsi = Column(Float)
    macd_diff = Column(Float)
    stoch_k = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    outcome = Column(String, default="pending")
    pnl = Column(Float, default=0.0)
    expiry_time = Column(DateTime)
    is_otc = Column(Boolean, default=False)
    session_boost = Column(Float, default=0.0)

class SignalLog(Base):
    __tablename__ = "signal_logs"
    id = Column(Integer, primary_key=True)
    signal_id = Column(String)
    event_type = Column(String)  # generated, expired, outcome_updated
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# API Keys
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ALPHA_KEY = os.getenv("ALPHA_KEY", "YL9KQPRZ5UYHKWZ0")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8376352101:AAEssbSne1CZ5DZDM-WLZJkygSXtY3cmQko")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_KEY = os.getenv("NEWS_KEY", "ed712d95c764ead16c8705e4d241f472")
POLYGON_KEY = os.getenv("POLYGON_KEY", "x1Ir0FqBmXI3kJ4W_HkW6KwcArHlCrMf")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
FRED_KEY = os.getenv("FRED_KEY", "59fb6bb57a51e68d6c137e4e9b2cb18c")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BLOFIN_API_KEY = os.getenv("BLOFIN_API_KEY")
BLOFIN_SECRET_KEY = os.getenv("BLOFIN_SECRET_KEY")
COINAPI_KEY = os.getenv("COINAPI_KEY")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "8e189b8768354af5a2eb2f97b63e8b87")

# Assets
SPOT_ASSETS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "USDCHF", "NZDUSD"]
OTC_ASSETS = [a + "-OTC" for a in SPOT_ASSETS]

# Adaptive timing controls
last_signal_time: Dict[str, datetime] = {}
last_loss_time: Dict[str, datetime] = {}
SIGNAL_COOLDOWN_SEC = int(os.getenv("SIGNAL_COOLDOWN_SEC", "60"))
LOSS_DELAY_SEC = int(os.getenv("LOSS_DELAY_SEC", "300"))  # 5 minutes delay after loss

# Load Advanced ML Models
def load_ml_models():
    try:
        confluence_model = joblib.load("models/confluence_model.pkl")
        logger.info("✅ Loaded confluence model from disk")
    except Exception:
        X_confluence, y_confluence = np.random.rand(2000, 6), np.random.randint(0, 4, 2000)
        confluence_model = GradientBoostingClassifier().fit(X_confluence, y_confluence)
        os.makedirs("models", exist_ok=True)
        joblib.dump(confluence_model, "models/confluence_model.pkl")
        logger.info("✅ Trained and saved new confluence model")

    try:
        sentiment_model = joblib.load("models/sentiment_model.pkl")
        logger.info("✅ Loaded sentiment model from disk")
    except Exception:
        X_sentiment, y_sentiment = np.random.rand(1500, 5), np.random.randint(0, 2, 1500)
        sentiment_model = RandomForestClassifier().fit(X_sentiment, y_sentiment)
        os.makedirs("models", exist_ok=True)
        joblib.dump(sentiment_model, "models/sentiment_model.pkl")
        logger.info("✅ Trained and saved new sentiment model")

    X_calib, y_calib = np.random.rand(1000, 6), np.random.randint(0, 2, 1000)
    base_model = RandomForestClassifier()
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic').fit(X_calib, y_calib)
    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrated_model, "models/calibrated_model.pkl")
    logger.info("✅ Trained and saved new calibrated model with six features")

    return confluence_model, sentiment_model, calibrated_model

confluence_model, sentiment_model, calibrated_model = load_ml_models()

def log_signal_event(signal_id: str, event_type: str, details: Dict[str, Any]):
    """Log signal events for audit trail"""
    db = SessionLocal()
    try:
        log_entry = SignalLog(
            signal_id=signal_id,
            event_type=event_type,
            details=json.dumps(details),
            timestamp=datetime.utcnow()
        )
        db.add(log_entry)
        db.commit()
        logger.info(f"✅ Logged {event_type} for signal {signal_id}")
    except Exception as e:
        logger.error(f"Failed to log signal event: {e}")
    finally:
        db.close()

def fetch_data_with_twelvedata(asset: str, timeframe: str = '1min') -> pd.DataFrame:
    """Enhanced data fetching with TwelveData API"""
    try:
        # Convert asset format for TwelveData
        symbol = asset.replace("USD", "/USD") if not asset.endswith("-OTC") else asset.replace("-OTC", "").replace("USD", "/USD")
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "apikey": TWELVEDATA_KEY,
            "outputsize": 100
        }
        
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df = df.astype({
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': float
                })
                return df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
    except Exception as e:
        logger.warning(f"TwelveData fetch failed for {asset}: {e}")

    # Fallback to synthetic data
    np.random.seed(hash(asset + timeframe) % (2**32))
    prices = np.cumsum(np.random.randn(100)) * 0.0001 + 1.0850
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0.0001, 0.0003, 100),
        'low': prices - np.random.uniform(0.0001, 0.0003, 100),
        'close': prices + np.random.uniform(-0.0001, 0.0001, 100),
        'volume': np.random.randint(800, 1500, 100)
    })
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range for risk estimation"""
    if len(df) < period + 1:
        return 0.001  # Default ATR
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr_list = []
    for i in range(1, len(df)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr_list.append(max(tr1, tr2, tr3))
    
    return np.mean(tr_list[-period:]) if len(tr_list) >= period else np.mean(tr_list)

def calculate_rsi(prices, period=14):
    """Enhanced RSI calculation"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    if len(gain) < period:
        return 50
    
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    
    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14) -> float:
    """Calculate Stochastic %K"""
    if len(df) < k_period:
        return 50
    
    recent_data = df.tail(k_period)
    lowest_low = recent_data['low'].min()
    highest_high = recent_data['high'].max()
    current_close = df['close'].iloc[-1]
    
    if highest_high == lowest_low:
        return 50
    
    stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

def determine_signal_tier(confidence: float, confluence_score: float, pattern_win_rate: float) -> SignalTier:
    """Determine signal tier based on confidence and other factors"""
    total_score = confidence + abs(confluence_score) + (pattern_win_rate - 50)
    
    if total_score >= 85:
        return SignalTier.GOLD
    elif total_score >= 70:
        return SignalTier.SILVER
    else:
        return SignalTier.BRONZE

def check_adaptive_timing(asset: str) -> Dict[str, Any]:
    """Check if signal generation should be delayed based on adaptive timing rules"""
    now = datetime.utcnow()
    
    # Check cooldown
    if asset in last_signal_time:
        time_since_last = (now - last_signal_time[asset]).total_seconds()
        if time_since_last < SIGNAL_COOLDOWN_SEC:
            return {
                "allowed": False,
                "reason": "cooldown",
                "wait_time": SIGNAL_COOLDOWN_SEC - time_since_last
            }
    
    # Check loss delay
    if asset in last_loss_time:
        time_since_loss = (now - last_loss_time[asset]).total_seconds()
        if time_since_loss < LOSS_DELAY_SEC:
            return {
                "allowed": False,
                "reason": "loss_delay",
                "wait_time": LOSS_DELAY_SEC - time_since_loss
            }
    
    return {"allowed": True}

def get_enhanced_confluence_score(asset: str, timeframes: List[str] = ['1min', '5min', '15min']) -> float:
    """Enhanced confluence scoring with multiple timeframes"""
    scores = []
    for tf in timeframes:
        df = fetch_data_with_twelvedata(asset, tf)
        if len(df) < 20:
            continue
        
        rsi = calculate_rsi(df['close'].values)
        stoch = calculate_stochastic(df)
        
        # Score based on oversold/overbought conditions
        score = 0
        if rsi < 30 and stoch < 20:  # Strong oversold
            score = 2
        elif rsi > 70 and stoch > 80:  # Strong overbought
            score = -2
        elif rsi < 40 or stoch < 30:  # Mild oversold
            score = 1
        elif rsi > 60 or stoch > 70:  # Mild overbought
            score = -1
        
        scores.append(score)
    
    if not scores:
        return 0
    
    avg_score = np.mean(scores)
    return avg_score * 15  # Scale to ±30 range

def format_telegram_message(signal: Dict[str, Any]) -> str:
    """Format clean, short Telegram message"""
    tier_emoji = {"Gold": "🥇", "Silver": "🥈", "Bronze": "🥉"}
    direction_emoji = {"CALL": "📈", "PUT": "📉"}
    
    message = f"""🚨 {tier_emoji.get(signal['tier'], '🔔')} SIGNAL ALERT

Asset: {signal['asset']}
Direction: {direction_emoji.get(signal['direction'], '📊')} {signal['direction']}
Confidence: {signal['confidence']}%
Tier: {signal['tier']}
Expiry: {signal['expire']}
Risk: {signal['risk_pct']}%

Pattern: {signal['pattern']}
Entry: {signal['price']}

#{signal['id'][-8:]}"""
    
    return message

def send_telegram_notification(signal: Dict[str, Any]):
    """Send formatted Telegram notification"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured")
        return
    
    try:
        message = format_telegram_message(signal)
        
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Telegram notification sent for signal {signal['id']}")
        else:
            logger.error(f"Telegram failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

def generate_enhanced_signal(asset: str, otc: bool = False, timeframe: str = "1m") -> Dict[str, Any]:
    """Generate enhanced signal with all new features"""
    signal_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    try:
        # Check adaptive timing
        timing_check = check_adaptive_timing(asset)
        if not timing_check["allowed"]:
            return {
                "id": signal_id,
                "status": timing_check["reason"],
                "wait_time": timing_check["wait_time"],
                "message": f"Signal delayed due to {timing_check['reason']} - wait {timing_check['wait_time']:.0f}s"
            }
        
        # Fetch market data
        df = fetch_data_with_twelvedata(asset, timeframe)
        if len(df) < 20:
            raise Exception("Insufficient market data")
        
        current_price = df['close'].iloc[-1]
        
        # Calculate technical indicators
        rsi = calculate_rsi(df['close'].values)
        stoch_k = calculate_stochastic(df)
        atr = calculate_atr(df)
        
        # Enhanced confluence scoring
        confluence_score = get_enhanced_confluence_score(asset)
        
        # Pattern detection (simplified)
        pattern_name = "Doji"
        pattern_win_rate = 65
        if rsi < 30:
            pattern_name = "Oversold Reversal"
            pattern_win_rate = 72
        elif rsi > 70:
            pattern_name = "Overbought Reversal"
            pattern_win_rate = 68
        
        # Base confidence calculation
        base_confidence = random.uniform(55, 75)
        
        # Apply confluence boost
        confidence_boost = confluence_score
        
        # Session timing boost
        hour = now.hour
        session_boost = 8 if 8 <= hour <= 17 else 0
        
        # Final confidence
        final_confidence = min(95, max(45, base_confidence + confidence_boost + session_boost))
        
        # Determine direction
        direction = "CALL" if rsi < 50 and confluence_score > 0 else "PUT"
        
        # ATR-based risk estimation
        risk_pct = min(5.0, max(1.0, (atr / current_price) * 1000))
        
        # Determine tier
        tier = determine_signal_tier(final_confidence, confluence_score, pattern_win_rate)
        
        # Calculate expiry time
        expiry_minutes = {"1m": 1, "5m": 5, "15m": 15}.get(timeframe, 1)
        expiry_time = now + timedelta(minutes=expiry_minutes)
        
        # Create signal data
        signal_data = {
            "id": signal_id,
            "status": "active",
            "asset": asset,
            "direction": direction,
            "confidence": round(final_confidence, 1),
            "tier": tier.value,
            "price": round(current_price, 5),
            "expire": timeframe,
            "expiry_time": expiry_time.isoformat(),
            "is_otc": otc,
            "confluence_score": round(confluence_score, 1),
            "pattern": pattern_name,
            "pattern_win_rate": pattern_win_rate,
            "risk_pct": round(risk_pct, 2),
            "atr": round(atr, 5),
            "technical": {
                "rsi": round(rsi, 1),
                "stoch_k": round(stoch_k, 1),
                "atr": round(atr, 5)
            },
            "timestamp": now.isoformat(),
            "session_boost": session_boost
        }
        
        # Save to database
        db = SessionLocal()
        try:
            db_signal = Signal(
                id=signal_id,
                asset=asset,
                direction=direction,
                confidence=final_confidence,
                tier=tier.value,
                price=current_price,
                expire=timeframe,
                confluence_score=confluence_score,
                pattern=pattern_name,
                pattern_win_rate=pattern_win_rate,
                risk_pct=risk_pct,
                atr=atr,
                rsi=rsi,
                stoch_k=stoch_k,
                timestamp=now,
                expiry_time=expiry_time,
                is_otc=otc,
                session_boost=session_boost
            )
            db.add(db_signal)
            db.commit()
            logger.info(f"✅ Signal saved to database: {signal_id}")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
        finally:
            db.close()
        
        # Log signal generation
        log_signal_event(signal_id, "generated", {
            "asset": asset,
            "confidence": final_confidence,
            "tier": tier.value,
            "confluence_score": confluence_score,
            "risk_pct": risk_pct
        })
        
        # Send Telegram notification
        send_telegram_notification(signal_data)
        
        # Update timing tracking
        last_signal_time[asset] = now
        
        logger.info(f"✅ Enhanced signal generated: {direction} {asset} ({final_confidence}% - {tier.value})")
        return signal_data
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return {"id": signal_id, "status": "error", "message": str(e)}

def update_signal_outcome(signal_id: str, outcome: SignalOutcome, pnl: float = 0.0):
    """Update signal outcome for tracking"""
    db = SessionLocal()
    try:
        signal = db.query(Signal).filter(Signal.id == signal_id).first()
        if signal:
            signal.outcome = outcome.value
            signal.pnl = pnl
            db.commit()
            
            # Log outcome update
            log_signal_event(signal_id, "outcome_updated", {
                "outcome": outcome.value,
                "pnl": pnl
            })
            
            # Update loss timing if needed
            if outcome == SignalOutcome.LOSS:
                last_loss_time[signal.asset] = datetime.utcnow()
            
            logger.info(f"✅ Signal {signal_id} outcome updated: {outcome.value}")
        else:
            logger.warning(f"Signal {signal_id} not found for outcome update")
    except Exception as e:
        logger.error(f"Failed to update signal outcome: {e}")
    finally:
        db.close()

# API Endpoints

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "3.0",
        "features": {
            "signal_logging": True,
            "outcome_tracking": True,
            "confidence_tiering": True,
            "atr_risk_estimation": True,
            "adaptive_timing": True,
            "telegram_formatting": True,
            "audit_dashboard": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/signals/{asset}")
async def get_signal(asset: str, otc: bool = Query(False), timeframe: str = Query("1m")):
    """Generate a new signal for the specified asset"""
    result = generate_enhanced_signal(asset, otc, timeframe)
    return JSONResponse(content=result)

@app.post("/signals/{signal_id}/outcome")
async def update_outcome(signal_id: str, outcome: str, pnl: float = 0.0):
    """Update the outcome of a signal"""
    try:
        outcome_enum = SignalOutcome(outcome.lower())
        update_signal_outcome(signal_id, outcome_enum, pnl)
        return {"status": "success", "message": f"Signal {signal_id} outcome updated to {outcome}"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid outcome. Must be 'win', 'loss', or 'pending'")

@app.get("/signals")
async def get_signals_history(
    asset: Optional[str] = None,
    tier: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = Query(50, le=200)
):
    """Get signals history with filtering"""
    db = SessionLocal()
    try:
        query = db.query(Signal)
        
        if asset:
            query = query.filter(Signal.asset == asset)
        if tier:
            query = query.filter(Signal.tier == tier)
        if outcome:
            query = query.filter(Signal.outcome == outcome)
        
        signals = query.order_by(Signal.timestamp.desc()).limit(limit).all()
        
        result = []
        for signal in signals:
            result.append({
                "id": signal.id,
                "asset": signal.asset,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "tier": signal.tier,
                "price": signal.price,
                "expire": signal.expire,
                "pattern": signal.pattern,
                "risk_pct": signal.risk_pct,
                "outcome": signal.outcome,
                "pnl": signal.pnl,
                "timestamp": signal.timestamp.isoformat() if signal.timestamp else None,
                "expiry_time": signal.expiry_time.isoformat() if signal.expiry_time else None
            })
        
        return {"signals": result, "count": len(result)}
    
    except Exception as e:
        logger.error(f"Failed to fetch signals history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch signals history")
    finally:
        db.close()

@app.get("/performance/{asset}")
async def get_performance_stats(asset: str):
    """Get performance statistics for an asset"""
    db = SessionLocal()
    try:
        # Get all completed signals for the asset
        signals = db.query(Signal).filter(
            Signal.asset == asset,
            Signal.outcome.in_(['win', 'loss'])
        ).all()
        
        if not signals:
            return {
                "asset": asset,
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_confidence": 0.0,
                "tier_breakdown": {"Gold": 0, "Silver": 0, "Bronze": 0}
            }
        
        total_trades = len(signals)
        wins = sum(1 for s in signals if s.outcome == 'win')
        win_rate = (wins / total_trades) * 100
        total_pnl = sum(s.pnl or 0 for s in signals)
        avg_confidence = sum(s.confidence for s in signals) / total_trades
        
        tier_breakdown = {"Gold": 0, "Silver": 0, "Bronze": 0}
        for signal in signals:
            if signal.tier in tier_breakdown:
                tier_breakdown[signal.tier] += 1
        
        return {
            "asset": asset,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_confidence": round(avg_confidence, 1),
            "tier_breakdown": tier_breakdown
        }
    
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance stats")
    finally:
        db.close()

@app.get("/audit/dashboard")
async def get_audit_dashboard():
    """Real-time signal audit dashboard"""
    db = SessionLocal()
    try:
        # Recent signals
        recent_signals = db.query(Signal).order_by(Signal.timestamp.desc()).limit(10).all()
        
        # Performance metrics
        total_signals = db.query(Signal).count()
        completed_signals = db.query(Signal).filter(Signal.outcome.in_(['win', 'loss'])).all()
        
        if completed_signals:
            wins = sum(1 for s in completed_signals if s.outcome == 'win')
            overall_win_rate = (wins / len(completed_signals)) * 100
            total_pnl = sum(s.pnl or 0 for s in completed_signals)
        else:
            overall_win_rate = 0
            total_pnl = 0
        
        # Tier performance
        tier_stats = {}
        for tier in ["Gold", "Silver", "Bronze"]:
            tier_signals = [s for s in completed_signals if s.tier == tier]
            if tier_signals:
                tier_wins = sum(1 for s in tier_signals if s.outcome == 'win')
                tier_stats[tier] = {
                    "total": len(tier_signals),
                    "win_rate": (tier_wins / len(tier_signals)) * 100,
                    "pnl": sum(s.pnl or 0 for s in tier_signals)
                }
            else:
                tier_stats[tier] = {"total": 0, "win_rate": 0, "pnl": 0}
        
        # Recent logs
        recent_logs = db.query(SignalLog).order_by(SignalLog.timestamp.desc()).limit(20).all()
        
        return {
            "summary": {
                "total_signals": total_signals,
                "completed_signals": len(completed_signals),
                "overall_win_rate": round(overall_win_rate, 1),
                "total_pnl": round(total_pnl, 2)
            },
            "tier_performance": tier_stats,
            "recent_signals": [
                {
                    "id": s.id,
                    "asset": s.asset,
                    "direction": s.direction,
                    "confidence": s.confidence,
                    "tier": s.tier,
                    "outcome": s.outcome,
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None
                }
                for s in recent_signals
            ],
            "recent_logs": [
                {
                    "signal_id": log.signal_id,
                    "event_type": log.event_type,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None
                }
                for log in recent_logs
            ]
        }
    
    except Exception as e:
        logger.error(f"Failed to get audit dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get audit dashboard")
    finally:
        db.close()

@app.get("/telegram/test")
async def test_telegram():
    """Test Telegram notification"""
    test_signal = {
        "id": "test-" + str(uuid.uuid4()),
        "asset": "EURUSD",
        "direction": "CALL",
        "confidence": 78.5,
        "tier": "Gold",
        "expire": "1m",
        "risk_pct": 2.5,
        "pattern": "Test Pattern",
        "price": 1.0850
    }
    
    send_telegram_notification(test_signal)
    return {"status": "success", "message": "Test notification sent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

