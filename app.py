#!/usr/bin/env python3
"""
Expert Financial Risk Analysis AI - Enhanced with Dark Theme and Advanced Visualizations
"""

import os
import sys
import subprocess
import shutil
import time
import uuid
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Check if this is the first run and setup is needed
def check_and_run_setup():
    """Check if setup is needed and run it automatically"""
    setup_marker = Path('.setup_complete')
    
    if not setup_marker.exists():
        print("🚀 First-time setup detected. Running automatic setup...")
        run_auto_setup()
        setup_marker.touch()
        print("✅ Setup completed! Restarting application...")
        time.sleep(2)

def run_auto_setup():
    """Automated setup and fix system"""
    print("=" * 60)
    print("🛠️  EXPERT FINANCIAL AI - AUTOMATIC SETUP")
    print("=" * 60)
    
    # Install core requirements
    print("\n📦 Installing core requirements...")
    core_packages = [
        "streamlit>=1.28.0", "openai>=1.3.0", "langchain>=0.1.0",
        "langchain-openai>=0.0.5", "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4", "pandas>=2.0.0", "numpy>=1.24.0",
        "plotly>=5.17.0", "yfinance>=0.2.18", "python-dotenv>=1.0.0",
        "duckduckgo-search>=3.9.0", "wbgapi>=1.0.12",
        "scikit-learn>=1.3.0", "scipy>=1.10.0"
    ]
    
    for package in core_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"⚠️  {package.split('>=')[0]} (install failed, continuing...)")
    
    # Create directory structure and files
    print("\n📁 Creating directory structure...")
    directories = ['config', 'services', 'utils', 'static', 'data', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    create_settings_file()
    create_env_file()
    create_sample_data()
    create_dark_css_file()
    
    print("\n🎉 Setup completed successfully!")

def create_settings_file():
    """Create settings.py file"""
    settings_content = '''"""Configuration settings"""
import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Expert Financial Risk Analysis AI"
VERSION = "2.0.0"
DEBUG = False
DEFAULT_LANGUAGE = "en"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMPANY_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "JNJ"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
COMMODITY_SYMBOLS = ["GC=F", "SI=F", "CL=F", "NG=F"]  # Gold, Silver, Oil, Natural Gas
FOREX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
CACHE_TTL_MEDIUM = 1800
AI_MODEL_CONFIG = {"PRIMARY_MODEL": "gpt-4", "TEMPERATURE": 0.1, "MAX_TOKENS": 4000}
'''
    
    with open('config/settings.py', 'w') as f:
        f.write(settings_content)
    print("✅ config/settings.py")

def create_env_file():
    """Create .env template file"""
    if not os.path.exists('.env'):
        env_content = '''OPENAI_API_KEY=your_openai_api_key_here
DEBUG=False
DEFAULT_LANGUAGE=en
'''
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env (template created)")

def create_sample_data():
    """Create sample CSV data"""
    try:
        import pandas as pd
        
        sample_data = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'JNJ'],
            'Company': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 
                       'Tesla Inc.', 'Meta Platforms', 'NVIDIA Corp.', 'JPMorgan Chase', 'Bank of America', 'Johnson & Johnson'],
            'Price': [175.50, 375.25, 2750.80, 3380.50, 250.75, 485.20, 875.30, 145.80, 32.45, 165.90],
            'PE_Ratio': [28.5, 32.1, 25.8, 58.2, 65.4, 22.8, 68.9, 12.5, 14.2, 16.8],
            'Market_Cap': [2800000, 2750000, 1850000, 1400000, 800000, 1200000, 2150000, 450000, 275000, 485000],
            'Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                      'Consumer Discretionary', 'Technology', 'Technology', 'Financial', 'Financial', 'Healthcare']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('financial_risk_analysis_large.csv', index=False)
        print("✅ Sample financial data CSV")
    except ImportError:
        print("⚠️  Pandas not available, skipping sample data creation")

def create_dark_css_file():
    """Create enhanced dark theme CSS file"""
    css_content = '''/* Enhanced Dark Theme for Financial AI */
    
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {
    --bg-primary: #0f1419;
    --bg-secondary: #1a1f29;
    --bg-tertiary: #252b37;
    --bg-card: #1e2329;
    --bg-question: #1a1f2e;
    --bg-answer: #1a2e1f;
    
    --text-primary: #ffffff;
    --text-secondary: #b8bcc8;
    --text-muted: #6c7293;
    
    --accent-blue: #4A90E2;
    --accent-green: #10b981;
    --accent-gold: #f59e0b;
    --accent-red: #ef4444;
    
    --border-primary: #2d3748;
    --border-secondary: #4a5568;
    
    --shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.3);
    --shadow-xl: 0 20px 40px rgba(0, 0, 0, 0.4);
}

/* Global dark theme override */
.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

/* Force all text to be white in dark theme */
* {
    color: #ffffff;
}

/* Override Streamlit's default text colors */
.stMarkdown, .stText, p, span, div {
    color: #ffffff !important;
}

/* Ensure markdown content is white */
.stMarkdown > div {
    color: #ffffff !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #ffffff !important;
}

/* Main header styling */
.main-header {
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 50%, #4a5568 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: var(--text-primary);
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-xl);
    border: 1px solid var(--border-primary);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(74, 144, 226, 0.1) 0%, transparent 100%);
    pointer-events: none;
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(45deg, #ffffff, #4A90E2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.main-header p {
    font-size: 1.3rem;
    font-weight: 500;
    opacity: 0.9;
    margin-top: 1rem;
}

/* Enhanced question box with dark theme */
.question-box {
    background: #1a1a1a !important;
    padding: 2rem;
    border-radius: 16px;
    border-left: 4px solid #4A90E2;
    margin: 1.5rem 0;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
    border: 1px solid #333;
}

.question-box h3 {
    color: #4A90E2 !important;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.question-box p {
    color: #ffffff !important;
    font-size: 1.2rem;
    font-weight: 500;
    line-height: 1.6;
    margin: 0;
}

/* Enhanced answer box with dark theme */
.answer-box {
    background: #1a1a1a !important;
    padding: 2rem;
    border-radius: 16px;
    border-left: 4px solid #10b981;
    margin: 1.5rem 0;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
    border: 1px solid #333;
}

.answer-box h3 {
    color: #10b981 !important;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.answer-box div {
    color: #ffffff !important;
    font-size: 1.1rem;
    line-height: 1.7;
    font-weight: 400;
}

.answer-box div * {
    color: #ffffff !important;
}

/* Override any Streamlit default styles */
.stMarkdown > div {
    color: #ffffff !important;
}

/* Force white text in all markdown content */
.question-box .stMarkdown,
.question-box .stMarkdown *,
.answer-box .stMarkdown,
.answer-box .stMarkdown * {
    color: #ffffff !important;
    background: transparent !important;
}

/* Section headers */
.section-header {
    color: var(--accent-blue);
    font-size: 2rem;
    font-weight: 700;
    margin: 2rem 0 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 3px solid var(--accent-blue);
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    position: relative;
}

.section-header::after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 0;
    width: 80px;
    height: 3px;
    background: var(--accent-gold);
    border-radius: 2px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-tertiary) 100%);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: var(--shadow-lg);
    text-align: center;
    margin: 1rem 0;
    border: 1px solid var(--border-primary);
    color: var(--text-primary);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-xl);
    border-color: var(--accent-blue);
}

/* Gold price special styling */
.gold-price-card {
    background: linear-gradient(135deg, #d4af37 0%, #b8860b 50%, #daa520 100%);
    padding: 2rem;
    border-radius: 16px;
    color: #000;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-xl);
    border: 2px solid #ffd700;
    position: relative;
    overflow: hidden;
}

.gold-price-card::before {
    content: '✨';
    position: absolute;
    top: 1rem;
    left: 1rem;
    font-size: 2rem;
    opacity: 0.7;
}

.gold-price-card::after {
    content: '✨';
    position: absolute;
    bottom: 1rem;
    right: 1rem;
    font-size: 2rem;
    opacity: 0.7;
}

/* Streamlit component overrides */
.stSelectbox > div > div {
    background-color: var(--bg-card) !important;
    border: 2px solid var(--border-primary) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.stTextArea > div > div > textarea {
    background-color: var(--bg-card) !important;
    border: 2px solid var(--border-primary) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue) 0%, #357abd 100%) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
    border-right: 1px solid var(--border-primary) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-card) !important;
    border: 2px solid var(--border-primary) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue) 0%, #357abd 100%) !important;
    color: var(--text-primary) !important;
    border-color: var(--accent-blue) !important;
}

/* Status indicators */
.status-active {
    color: var(--accent-green);
    font-weight: 600;
}

.status-inactive {
    color: var(--accent-red);
    font-weight: 600;
}

/* Animation effects */
@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px var(--accent-blue); }
    50% { box-shadow: 0 0 20px var(--accent-blue), 0 0 30px var(--accent-blue); }
}

.glow-effect {
    animation: glow 2s ease-in-out infinite;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.2rem;
    }
    
    .question-box, .answer-box {
        padding: 1.5rem;
        margin: 1rem 0;
    }
}
'''
    
    Path('static').mkdir(exist_ok=True)
    with open('static/styles.css', 'w') as f:
        f.write(css_content)
    print("✅ static/styles.css (Dark Theme)")

# Run setup check
check_and_run_setup()

# Import required modules
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Import settings
    try:
        from config.settings import *
    except ImportError:
        try:
            exec(open('config/settings.py').read()) if os.path.exists('config/settings.py') else None
            if 'APP_NAME' not in locals():
                APP_NAME = "Expert Financial Risk Analysis AI"
                VERSION = "2.0.0"
                DEBUG = False
                DEFAULT_LANGUAGE = "en"
                OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                COMPANY_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "JNJ"]
                CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
                COMMODITY_SYMBOLS = ["GC=F", "SI=F", "CL=F", "NG=F"]
                FOREX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
                CACHE_TTL_MEDIUM = 1800
                AI_MODEL_CONFIG = {"PRIMARY_MODEL": "gpt-4", "TEMPERATURE": 0.1, "MAX_TOKENS": 4000}
        except:
            # Fallback settings if config file is not available
            APP_NAME = "Expert Financial Risk Analysis AI"
            VERSION = "2.0.0"
            DEBUG = False
            DEFAULT_LANGUAGE = "en"
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            COMPANY_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "JNJ"]
            CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
            COMMODITY_SYMBOLS = ["GC=F", "SI=F", "CL=F", "NG=F"]
            FOREX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
            CACHE_TTL_MEDIUM = 1800
            AI_MODEL_CONFIG = {"PRIMARY_MODEL": "gpt-4", "TEMPERATURE": 0.1, "MAX_TOKENS": 4000}
            
    # Ensure all required variables are available globally
    if 'CRYPTO_SYMBOLS' not in globals():
        CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
    if 'COMMODITY_SYMBOLS' not in globals():
        COMMODITY_SYMBOLS = ["GC=F", "SI=F", "CL=F", "NG=F"]
    if 'FOREX_SYMBOLS' not in globals():
        FOREX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]

except ImportError as e:
    st.error(f"Required packages not installed. Please run: pip install streamlit pandas numpy plotly yfinance")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with dark theme
st.set_page_config(
    page_title=f"🤖 {APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Enhanced Financial Data Fetcher with Online APIs
class AdvancedDataFetcher:
    def __init__(self):
        # Use global variables with fallbacks
        self.company_symbols = globals().get('COMPANY_SYMBOLS', ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
        self.crypto_symbols = globals().get('CRYPTO_SYMBOLS', ["BTC-USD", "ETH-USD", "BNB-USD"])
        self.commodity_symbols = globals().get('COMMODITY_SYMBOLS', ["GC=F", "SI=F", "CL=F"])
        self.forex_symbols = globals().get('FOREX_SYMBOLS', ["EURUSD=X", "GBPUSD=X"])
    
    def fetch_gold_data(self):
        """Fetch online gold data using yfinance as requested"""
        try:
            # Fetch historical data for Gold Futures
            gold = yf.Ticker("GC=F")
            hist = gold.history(period="1mo")
            
            if hist.empty:
                return None
            
            # Get current price
            try:
                current_price = gold.info.get('regularMarketPrice')
                if not current_price:
                    current_price = hist['Close'].iloc[-1]
            except:
                current_price = hist['Close'].iloc[-1]
            
            # Calculate additional metrics
            daily_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            weekly_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-7]) / hist['Close'].iloc[-7]) * 100 if len(hist) >= 7 else 0
            monthly_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            
            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
            
            return {
                'current_price': current_price,
                'daily_change': daily_change,
                'weekly_change': weekly_change,
                'monthly_change': monthly_change,
                'volatility': volatility,
                'history': hist,
                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0,
                'high_52w': hist['High'].max(),
                'low_52w': hist['Low'].min()
            }
        except Exception as e:
            logger.error(f"Error fetching gold data: {e}")
            return None
    
    def fetch_comprehensive_market_data(self):
        """Fetch comprehensive market data from multiple sources"""
        data = {}
        
        # Stocks
        data['stocks'] = self.fetch_stocks_data()
        
        # Cryptocurrencies
        data['crypto'] = self.fetch_crypto_data()
        
        # Commodities
        data['commodities'] = self.fetch_commodities_data()
        
        # Forex
        data['forex'] = self.fetch_forex_data()
        
        # Gold (special handling as requested)
        data['gold'] = self.fetch_gold_data()
        
        # Market indices
        data['indices'] = self.fetch_market_indices()
        
        # CSV data
        try:
            if os.path.exists('financial_risk_analysis_large.csv'):
                data['csv_data'] = pd.read_csv('financial_risk_analysis_large.csv')
        except Exception as e:
            logger.warning(f"Error loading CSV data: {e}")
        
        return data
    
    def fetch_stocks_data(self):
        """Fetch stocks data"""
        stocks_data = {}
        
        for symbol in self.company_symbols:
            try:
                ticker = yf.Ticker(symbol)
                # IMPROVED: Extended period for better correlation analysis
                hist = ticker.history(period="1y")  # Increased from 6mo to 1y
                info = ticker.info
                
                if not hist.empty:
                    # IMPROVED: Better data validation and cleaning
                    hist = hist.dropna()  # Remove NaN values immediately
                    
                    if len(hist) > 30:  # Ensure sufficient data points (at least 30 days)
                        stocks_data[symbol] = {
                            'history': hist,  # Now contains cleaned data
                            'info': info,
                            'current_price': hist['Close'].iloc[-1],
                            'daily_change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0,
                            'volume': hist['Volume'].iloc[-1] if 'Volume' in hist and not hist['Volume'].empty else 0
                        }
            except Exception as e:
                logger.warning(f"Error fetching stock {symbol}: {e}")
                continue
        
        return stocks_data
    
    def fetch_crypto_data(self):
        """Fetch cryptocurrency data"""
        crypto_data = {}
        
        for symbol in self.crypto_symbols:
            try:
                ticker = yf.Ticker(symbol)
                # IMPROVED: Extended period and better error handling for crypto
                hist = ticker.history(period="6mo")  # Increased from 3mo to 6mo for better correlation data
                
                if not hist.empty:
                    # IMPROVED: Better data validation and cleaning
                    hist = hist.dropna()  # Remove NaN values immediately
                    
                    if len(hist) > 10:  # Ensure sufficient data points
                        daily_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
                        
                        crypto_data[symbol] = {
                            'current_price': hist['Close'].iloc[-1],
                            'daily_change': daily_change,
                            'history': hist,  # Now contains cleaned data
                            'volume_24h': hist['Volume'].iloc[-1] if 'Volume' in hist and not hist['Volume'].empty else 0
                        }
            except Exception as e:
                logger.warning(f"Error fetching crypto {symbol}: {e}")
                continue
        
        return crypto_data
    
    def fetch_commodities_data(self):
        """Fetch commodities data"""
        commodities_data = {}
        
        commodity_names = {
            "GC=F": "Gold",
            "SI=F": "Silver", 
            "CL=F": "Crude Oil",
            "NG=F": "Natural Gas"
        }
        
        for symbol in self.commodity_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
                
                if not hist.empty:
                    daily_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
                    
                    commodities_data[symbol] = {
                        'name': commodity_names.get(symbol, symbol),
                        'current_price': hist['Close'].iloc[-1],
                        'daily_change': daily_change,
                        'history': hist
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                continue
        
        return commodities_data
    
    def fetch_forex_data(self):
        """Fetch forex data"""
        forex_data = {}
        
        for symbol in self.forex_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                
                if not hist.empty:
                    daily_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
                    
                    forex_data[symbol] = {
                        'current_rate': hist['Close'].iloc[-1],
                        'daily_change': daily_change,
                        'history': hist
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                continue
        
        return forex_data
    
    def fetch_market_indices(self):
        """Fetch major market indices"""
        indices_data = {}
        indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]  # S&P 500, Dow Jones, NASDAQ, VIX
        
        for symbol in indices:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                
                if not hist.empty:
                    daily_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
                    
                    indices_data[symbol] = {
                        'current_value': hist['Close'].iloc[-1],
                        'daily_change': daily_change,
                        'history': hist
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                continue
        
        return indices_data

# Enhanced Visualizations with Dark Theme
class EnhancedFinancialVisualizations:
    def __init__(self):
        # Dark theme color palette
        self.dark_colors = {
            'primary': '#4A90E2',
            'secondary': '#10b981',
            'accent': '#f59e0b',
            'danger': '#ef4444',
            'success': '#10b981',
            'warning': '#f59e0b',
            'info': '#4A90E2',
            'background': '#1a1f29',
            'surface': '#252b37'
        }
        
        # Dark theme template for all charts
        self.dark_template = {
            'layout': {
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'colorway': ['#4A90E2', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'],
                'font': {'color': '#ffffff', 'family': 'Inter'},
                'xaxis': {
                    'gridcolor': '#2d3748',
                    'linecolor': '#4a5568',
                    'tickcolor': '#4a5568',
                    'zerolinecolor': '#4a5568',
                    'color': '#ffffff'
                },
                'yaxis': {
                    'gridcolor': '#2d3748',
                    'linecolor': '#4a5568',
                    'tickcolor': '#4a5568',
                    'zerolinecolor': '#4a5568',
                    'color': '#ffffff'
                },
                'title': {'font': {'color': '#ffffff', 'size': 18}}
            }
        }
    
    def apply_dark_theme(self, fig):
        """Apply dark theme to any plotly figure"""
        fig.update_layout(self.dark_template['layout'])
        return fig
    
    def create_gold_price_chart(self, gold_data):
        """Create enhanced gold price chart"""
        if not gold_data or 'history' not in gold_data:
            return None
        
        hist = gold_data['history']
        
        # Create candlestick chart for gold
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Gold Price",
            increasing_line_color='#d4af37',
            decreasing_line_color='#b8860b'
        )])
        
        # Add moving averages
        if len(hist) >= 20:
            ma_20 = hist['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ma_20,
                mode='lines',
                name='MA 20',
                line=dict(color='#ffd700', width=2)
            ))
        
        fig.update_layout(
            title=f"🥇 Gold Price (GC=F) - Current: ${gold_data['current_price']:.2f}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500
        )
        
        return self.apply_dark_theme(fig)
    
    def create_comprehensive_market_dashboard(self, market_data):
        """Create comprehensive market overview with multiple asset classes"""
        if not market_data:
            return None
        
        # Create subplots for different asset classes
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['📈 Stock Performance', '₿ Cryptocurrency', '🛢️ Commodities', '💱 Forex'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Stocks subplot
        if 'stocks' in market_data and market_data['stocks']:
            for i, (symbol, data) in enumerate(list(market_data['stocks'].items())[:5]):
                if 'history' in data and not data['history'].empty:
                    hist = data['history']
                    normalized_prices = (hist['Close'] / hist['Close'].iloc[0]) * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=normalized_prices,
                            mode='lines',
                            name=symbol,
                            line=dict(width=3)
                        ),
                        row=1, col=1
                    )
        
        # Crypto subplot
        if 'crypto' in market_data and market_data['crypto']:
            for symbol, data in list(market_data['crypto'].items())[:3]:
                if 'history' in data and not data['history'].empty:
                    hist = data['history']
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name=symbol.replace('-USD', ''),
                            line=dict(width=3)
                        ),
                        row=1, col=2
                    )
        
        # Commodities subplot
        if 'commodities' in market_data and market_data['commodities']:
            for symbol, data in market_data['commodities'].items():
                if 'history' in data and not data['history'].empty:
                    hist = data['history']
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name=data.get('name', symbol),
                            line=dict(width=3)
                        ),
                        row=2, col=1
                    )
        
        # Forex subplot
        if 'forex' in market_data and market_data['forex']:
            for symbol, data in market_data['forex'].items():
                if 'history' in data and not data['history'].empty:
                    hist = data['history']
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name=symbol.replace('=X', ''),
                            line=dict(width=3)
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title="🌍 Global Financial Markets Overview",
            height=700,
            showlegend=True
        )
        
        return self.apply_dark_theme(fig)
    
    def create_sector_performance_chart(self, stocks_data):
        """Create sector performance analysis"""
        if not stocks_data:
            return None
        
        # Group stocks by sector
        sectors = {}
        for symbol, data in stocks_data.items():
            if 'info' in data:
                sector = data['info'].get('sector', 'Unknown')
                if sector not in sectors:
                    sectors[sector] = {'symbols': [], 'performance': [], 'market_cap': 0}
                
                sectors[sector]['symbols'].append(symbol)
                sectors[sector]['performance'].append(data.get('daily_change', 0))
                sectors[sector]['market_cap'] += data['info'].get('marketCap', 0)
        
        # Calculate average performance per sector
        sector_names = []
        avg_performance = []
        total_market_caps = []
        
        for sector, info in sectors.items():
            if info['performance']:
                sector_names.append(sector)
                avg_performance.append(np.mean(info['performance']))
                total_market_caps.append(info['market_cap'] / 1e9)  # Convert to billions
        
        if not sector_names:
            return None
        
        # Create bubble chart
        fig = go.Figure(data=go.Scatter(
            x=sector_names,
            y=avg_performance,
            mode='markers',
            marker=dict(
                size=[cap/50 for cap in total_market_caps],
                color=avg_performance,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Performance (%)", font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                line=dict(width=2, color='white'),
                sizemode='diameter'
            ),
            text=[f'{sector}<br>Avg Performance: {perf:.2f}%<br>Market Cap: ${cap:.1f}B' 
                  for sector, perf, cap in zip(sector_names, avg_performance, total_market_caps)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="📊 Sector Performance Analysis",
            xaxis_title="Sector",
            yaxis_title="Average Daily Performance (%)",
            height=500
        )
        
        return self.apply_dark_theme(fig)
    
    def create_crypto_performance_chart(self, crypto_data):
        """Create cryptocurrency performance chart"""
        if not crypto_data:
            return None
        
        symbols = []
        prices = []
        changes = []
        volumes = []
        
        for symbol, data in crypto_data.items():
            symbols.append(symbol.replace('-USD', ''))
            prices.append(data['current_price'])
            changes.append(data['daily_change'])
            volumes.append(data.get('volume_24h', 0))
        
        if not symbols:
            return None
        
        # Create subplot with price and volume
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['💰 Cryptocurrency Prices', '📊 24h Volume'],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        colors = ['#f59e0b' if change >= 0 else '#ef4444' for change in changes]
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=prices,
                name='Price',
                marker_color=colors,
                text=[f'${price:,.2f}<br>{change:+.2f}%' for price, change in zip(prices, changes)],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=volumes,
                name='Volume',
                marker_color='rgba(74, 144, 226, 0.7)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="₿ Cryptocurrency Market Analysis",
            height=600
        )
        
        return self.apply_dark_theme(fig)
    
    def create_market_correlation_heatmap(self, market_data):
        """Create correlation heatmap across asset classes"""
        if not market_data:
            return None
        
        # Collect price data from different asset classes
        price_data = {}
        
        # Add stocks
        if 'stocks' in market_data:
            for symbol, data in list(market_data['stocks'].items())[:8]:
                if 'history' in data and not data['history'].empty:
                    # IMPROVED: Add data validation and cleaning
                    close_prices = data['history']['Close'].dropna()  # Remove NaN values
                    if len(close_prices) > 10:  # Ensure sufficient data points
                        price_data[symbol] = close_prices
        
        # Add crypto
        if 'crypto' in market_data:
            for symbol, data in list(market_data['crypto'].items())[:3]:
                if 'history' in data and not data['history'].empty:
                    # IMPROVED: Add data validation and cleaning for crypto
                    close_prices = data['history']['Close'].dropna()  # Remove NaN values
                    if len(close_prices) > 10:  # Ensure sufficient data points
                        crypto_name = symbol.replace('-USD', '')
                        price_data[crypto_name] = close_prices
        
        # Add commodities
        if 'commodities' in market_data:
            for symbol, data in market_data['commodities'].items():
                if 'history' in data and not data['history'].empty:
                    # IMPROVED: Add data validation and cleaning for commodities
                    close_prices = data['history']['Close'].dropna()  # Remove NaN values
                    if len(close_prices) > 10:  # Ensure sufficient data points
                        price_data[data.get('name', symbol)] = close_prices
        
        if len(price_data) < 2:
            return None
        
        # IMPROVED: Create DataFrame with proper date alignment and handle missing data
        try:
            df = pd.DataFrame(price_data)
            
            # FIXED: Handle NaN values properly before correlation calculation
            # Method 1: Drop rows where any asset has NaN
            df_clean = df.dropna()
            
            # If too much data is lost, use forward fill + backward fill
            if len(df_clean) < len(df) * 0.5:  # If we lose more than 50% of data
                df_clean = df.fillna(method='ffill').fillna(method='bfill')
                df_clean = df_clean.dropna()  # Remove any remaining NaN
            
            # Ensure we still have enough data
            if len(df_clean) < 10:
                st.warning("Insufficient data for correlation analysis after cleaning")
                return None
            
            # Calculate correlation matrix with cleaned data
            correlation_matrix = df_clean.corr()
            
            # ADDITIONAL SAFETY: Replace any remaining NaN with 0
            correlation_matrix = correlation_matrix.fillna(0)
            
        except Exception as e:
            st.error(f"Error processing correlation data: {e}")
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "white"},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🔗 Cross-Asset Correlation Matrix (Cleaned Data)",  # IMPROVED: Updated title
            height=600
        )
        
        return self.apply_dark_theme(fig)

# Enhanced AI Agent with Real-time Data Fetching
class EnhancedFinancialAgent:
    def __init__(self):
        self.conversation_history = []
        self.initialized = False
        self.data_fetcher = AdvancedDataFetcher()
        
        if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
            try:
                import openai
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                self.initialized = True
            except ImportError:
                st.warning("OpenAI package not installed. Install with: pip install openai")
            except Exception as e:
                st.error(f"Error initializing OpenAI client: {e}")
    
    def detect_data_request(self, question: str) -> Dict[str, Any]:
        """Detect if user is asking for real-time data"""
        question_lower = question.lower()
        data_requests = {}
        
        # Gold price detection
        if any(word in question_lower for word in ['gold', 'ذهب', 'price', 'سعر']):
            data_requests['gold'] = True
        
        # Stock price detection
        stock_symbols = ['aapl', 'apple', 'msft', 'microsoft', 'googl', 'google', 'amzn', 'amazon', 'tsla', 'tesla']
        if any(symbol in question_lower for symbol in stock_symbols):
            data_requests['stocks'] = True
        
        # Crypto detection
        if any(word in question_lower for word in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'عملة رقمية']):
            data_requests['crypto'] = True
        
        # Market data detection
        if any(word in question_lower for word in ['market', 'dow', 'nasdaq', 's&p', 'index', 'سوق']):
            data_requests['indices'] = True
        
        # Current/now/today detection
        if any(word in question_lower for word in ['current', 'now', 'today', 'latest', 'الآن', 'اليوم', 'حالياً']):
            data_requests['real_time'] = True
        
        return data_requests
    
    def fetch_real_time_data(self, data_requests: Dict[str, Any]) -> str:
        """Fetch real-time data based on user request"""
        data_summary = []
        
        try:
            # IMPROVED: Enhanced gold data fetching with fallback
            if data_requests.get('gold') or data_requests.get('real_time'):
                gold_data = self.data_fetcher.fetch_gold_data()
                if gold_data:
                    # IMPROVED: Structured and formatted output
                    data_summary.append(f"""
📊 **LIVE GOLD MARKET ANALYSIS**
═══════════════════════════════════════════
💰 **Current Price:** ${gold_data['current_price']:.2f} per troy ounce
📈 **Daily Change:** {gold_data['daily_change']:+.2f}%
📅 **Weekly Change:** {gold_data['weekly_change']:+.2f}%
🗓️  **Monthly Change:** {gold_data['monthly_change']:+.2f}%
⬆️  **52-Week High:** ${gold_data['high_52w']:.2f}
⬇️  **52-Week Low:** ${gold_data['low_52w']:.2f}
📊 **Volatility:** {gold_data['volatility']:.1f}%
🔄 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════
                    """)
                else:
                    # IMPROVED: Fallback when API fails
                    data_summary.append("""
⚠️ **GOLD DATA UNAVAILABLE**
API data temporarily unavailable. Based on recent market trends,
gold typically trades between $1,900-$2,100 per ounce with 
daily volatility of 1-3%. Please check financial news sources
for the most current pricing.
                    """)
            
            # IMPROVED: Enhanced stock data with better formatting
            if data_requests.get('stocks') or data_requests.get('real_time'):
                stocks_data = self.data_fetcher.fetch_stocks_data()
                if stocks_data:
                    stock_summary = """
📈 **LIVE STOCK MARKET DATA**
═══════════════════════════════════════════
"""
                    for symbol, data in list(stocks_data.items())[:5]:
                        change_icon = "📈" if data['daily_change'] >= 0 else "📉"
                        stock_summary += f"• **{symbol}:** ${data['current_price']:.2f} {change_icon} {data['daily_change']:+.2f}%\n"
                    
                    stock_summary += f"""
🔄 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════
"""
                    data_summary.append(stock_summary)
                else:
                    # IMPROVED: Fallback for stocks
                    data_summary.append("""
⚠️ **STOCK DATA UNAVAILABLE**
Market data temporarily unavailable. Major indices typically
show daily movements of ±1-3%. Please check financial platforms
like Yahoo Finance or Bloomberg for current prices.
                    """)
            
            # IMPROVED: Enhanced crypto data with better formatting
            if data_requests.get('crypto') or data_requests.get('real_time'):
                crypto_data = self.data_fetcher.fetch_crypto_data()
                if crypto_data:
                    crypto_summary = """
₿ **LIVE CRYPTOCURRENCY MARKET**
═══════════════════════════════════════════
"""
                    for symbol, data in list(crypto_data.items())[:3]:
                        crypto_name = symbol.replace('-USD', '')
                        change_icon = "📈" if data['daily_change'] >= 0 else "📉"
                        crypto_summary += f"• **{crypto_name}:** ${data['current_price']:,.2f} {change_icon} {data['daily_change']:+.2f}%\n"
                    
                    crypto_summary += f"""
🔄 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════
"""
                    data_summary.append(crypto_summary)
                else:
                    # IMPROVED: Fallback for crypto
                    data_summary.append("""
⚠️ **CRYPTO DATA UNAVAILABLE**
Cryptocurrency data temporarily unavailable. Bitcoin typically
trades with high volatility (5-15% daily swings). Check 
CoinMarketCap or CoinGecko for real-time prices.
                    """)
            
            # IMPROVED: Enhanced market indices with better formatting
            if data_requests.get('indices') or data_requests.get('real_time'):
                indices_data = self.data_fetcher.fetch_market_indices()
                if indices_data:
                    indices_summary = """
📊 **LIVE MARKET INDICES**
═══════════════════════════════════════════
"""
                    index_names = {
                        "^GSPC": "S&P 500",
                        "^DJI": "Dow Jones",
                        "^IXIC": "NASDAQ",
                        "^VIX": "VIX (Fear Index)"
                    }
                    for symbol, data in indices_data.items():
                        name = index_names.get(symbol, symbol)
                        change_icon = "📈" if data['daily_change'] >= 0 else "📉"
                        indices_summary += f"• **{name}:** {data['current_value']:.2f} {change_icon} {data['daily_change']:+.2f}%\n"
                    
                    indices_summary += f"""
🔄 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════
"""
                    data_summary.append(indices_summary)
                else:
                    # IMPROVED: Fallback for indices
                    data_summary.append("""
⚠️ **MARKET INDICES UNAVAILABLE**
Index data temporarily unavailable. Major US indices typically
show daily movements of ±0.5-2%. Check financial news sources
for current market conditions.
                    """)
        
        except Exception as e:
            # IMPROVED: Better error handling with user-friendly message
            data_summary.append(f"""
🚨 **DATA FETCH ERROR**
═══════════════════════════════════════════
An error occurred while fetching real-time data: {str(e)}

**Recommended Actions:**
• Check your internet connection
• Verify API keys are properly configured
• Try refreshing the data in a few minutes
• Use alternative financial data sources temporarily
═══════════════════════════════════════════
            """)
        
        return "\n".join(data_summary) if data_summary else ""
    
    def process_question(self, question: str, user_context: str = "") -> str:
        """Enhanced question processing with real-time data fetching"""
        if not self.initialized:
            return "AI Agent is not available. Please check your OpenAI API key configuration."
        
        try:
            # Detect if user wants real-time data
            data_requests = self.detect_data_request(question)
            real_time_data = ""
            
            # Fetch real-time data if requested
            if data_requests:
                with st.spinner("🔄 Fetching real-time market data..."):
                    real_time_data = self.fetch_real_time_data(data_requests)
            
            # Enhanced system message with real-time data capability
            system_message = f"""You are an Expert Financial AI Assistant with access to real-time market data. 

**Your Capabilities:**
- Investment Analysis & Portfolio Management
- Risk Assessment & Management  
- Financial Planning & Strategy
- Market Analysis & Economics
- Corporate Finance & Valuation
- Cryptocurrency and Digital Assets Analysis
- Commodities and Precious Metals Trading
- Forex and International Markets
- **REAL-TIME DATA ACCESS** for gold, stocks, crypto, and market indices

**Current Real-Time Market Data:**
{real_time_data}

            **Instructions:**
- Use the real-time data provided above in your analysis
- Always mention current prices when discussing investments
- Provide comprehensive, professional financial advice in a CLEAN, STRUCTURED FORMAT
- Use clear sections with headers, bullet points, and proper spacing
- Consider risk factors and market conditions
- Be specific and actionable in your recommendations
- Support your analysis with the current market data provided
- If asked about gold prices specifically, use the live gold data provided
- For Arabic questions, respond in Arabic with proper formatting
- ALWAYS format your response with clear sections and proper spacing

**Response Format Guidelines:**
- Use clear headers (## for main sections, ### for subsections)
- Use bullet points and numbered lists where appropriate
- Include proper spacing between sections
- Highlight key numbers and percentages
- Provide structured recommendations with clear action items

**Important:** The real-time data above is fetched live from Yahoo Finance and is current as of now."""
            
            # Prepare the full prompt
            full_prompt = f"{user_context}\n\nQuestion: {question}" if user_context else question
            
            # Make API call
            response = self.client.chat.completions.create(
                model=AI_MODEL_CONFIG.get("PRIMARY_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=AI_MODEL_CONFIG.get("TEMPERATURE", 0.1),
                max_tokens=AI_MODEL_CONFIG.get("MAX_TOKENS", 4000)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in AI processing: {e}")
            return f"I encountered an error while processing your question. Please try again or check your API configuration. Error: {str(e)}"

class SimpleUserProfileManager:
    def collect_and_process_user_info(self, question: str) -> str:
        """Extract user information from question"""
        context_parts = []
        
        # Detect language
        arabic_chars = len([c for c in question if '\u0600' <= c <= '\u06FF'])
        if arabic_chars > len(question) * 0.3:
            context_parts.append("Language: Arabic")
        else:
            context_parts.append("Language: English")
        
        # Detect asset preferences
        if any(word in question.lower() for word in ['gold', 'ذهب']):
            context_parts.append("Interest: Gold/Precious Metals")
        if any(word in question.lower() for word in ['crypto', 'bitcoin', 'عملة رقمية']):
            context_parts.append("Interest: Cryptocurrency")
        if any(word in question.lower() for word in ['stock', 'share', 'أسهم']):
            context_parts.append("Interest: Stock Market")
        
        # Risk tolerance
        if any(word in question.lower() for word in ['conservative', 'safe', 'low risk', 'محافظ']):
            context_parts.append("Risk Tolerance: Conservative")
        elif any(word in question.lower() for word in ['aggressive', 'high risk', 'growth', 'عدواني']):
            context_parts.append("Risk Tolerance: Aggressive")
        
        return "User Context: " + "; ".join(context_parts) if context_parts else ""
    
    def display_profile_summary(self):
        """Display profile summary"""
        profile = st.session_state.get('user_profile', {})
        if profile:
            for key, value in profile.items():
                st.write(f"**{key}:** {value}")
        else:
            st.info("🔍 Profile information is extracted automatically from your questions.")

# Load CSS
def load_css():
    """Load enhanced dark theme CSS"""
    try:
        css_path = 'static/styles.css'
        if os.path.exists(css_path):
            with open(css_path, 'r') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        # Fallback dark theme
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f29 100%); color: white; }
        .question-box { background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3a 100%); 
                       padding: 2rem; border-radius: 16px; border-left: 4px solid #4A90E2; 
                       margin: 1.5rem 0; color: white; }
        .answer-box { background: linear-gradient(135deg, #1a2e1f 0%, #2a3a2f 100%); 
                     padding: 2rem; border-radius: 16px; border-left: 4px solid #10b981; 
                     margin: 1.5rem 0; color: white; }
        </style>
        """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'conversation_history': [],
        'conversation_id': str(uuid.uuid4()),
        'user_profile': {},
        'app_settings': {'language': DEFAULT_LANGUAGE},
        'current_data': {},
        'services_initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def initialize_services():
    """Initialize services"""
    return {
        'ai_agent': EnhancedFinancialAgent(),
        'user_manager': SimpleUserProfileManager(),
        'data_fetcher': AdvancedDataFetcher(),
        'visualizer': EnhancedFinancialVisualizations()
    }

def display_header():
    """Display header"""
    lang = st.session_state.app_settings.get('language', 'en')
    
    if lang == 'ar':
        title = "🤖 مساعد الذكي المالي المتقدم"
        subtitle = "تمكين بناء الثروة من خلال الرؤى الفورية والتحليل المخصص"
    else:
        title = f"🤖 {APP_NAME}"
        subtitle = "Empowering Wealth Creation with Real-time Insights & Personalized Analysis"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <small>Version {VERSION} | Enhanced Dark Theme | Real-time Global Markets</small>
    </div>
    """, unsafe_allow_html=True)

def create_chat_interface(services):
    """Create enhanced chat interface"""
    st.markdown('<h2 class="section-header">💬 Ask Your Financial AI Agent</h2>', unsafe_allow_html=True)
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        st.error("🔑 Please set your OPENAI_API_KEY in the .env file")
        st.code("OPENAI_API_KEY=sk-your-actual-key-here", language="bash")
        return
    
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Ask me anything about finance, markets, investments, or economics:",
            placeholder="e.g., What's the current gold price trend and should I invest now?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit = st.form_submit_button("🚀 Ask Agent", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
    
    if clear:
        st.session_state.conversation_history = []
        st.success("Chat cleared!")
        st.rerun()
    
    if question and submit:
        handle_question(question, services)

def handle_question(question: str, services):
    """Handle user question with enhanced dark theme styling"""
    # Display question with forced dark theme
    st.markdown(f"""
    <div style="
        background: #1a1a1a !important;
        padding: 2rem;
        border-radius: 16px;
        border-left: 4px solid #4A90E2;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        border: 1px solid #333;
    ">
        <h3 style="color: #4A90E2 !important; font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem;">
            ❓ Your Question:
        </h3>
        <p style="color: #ffffff !important; font-size: 1.2rem; font-weight: 500; line-height: 1.6; margin: 0;">
            {question}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show progress
    with st.spinner("🤖 Processing your question with real-time market data..."):
        user_context = services['user_manager'].collect_and_process_user_info(question)
        response = services['ai_agent'].process_question(question, user_context)
    
    # Display response with forced dark theme
    st.markdown(f"""
    <div style="
        background: #1a1a1a !important;
        padding: 2rem;
        border-radius: 16px;
        border-left: 4px solid #10b981;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        border: 1px solid #333;
    ">
        <h3 style="color: #10b981 !important; font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem;">
            🤖 AI Agent Response:
        </h3>
        <div style="color: #ffffff !important; font-size: 1.1rem; line-height: 1.7; font-weight: 400;">
            {response}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Save to history
    st.session_state.conversation_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': response
    })

def create_enhanced_dashboard(services, data):
    """Create enhanced dashboard with all requested features"""
    st.markdown('<h2 class="section-header">📊 Real-time Global Financial Markets</h2>', unsafe_allow_html=True)
    
    if not data:
        st.warning("No financial data available. Please check your internet connection.")
        return
    
    # Gold Price Section (as specifically requested)
    if 'gold' in data and data['gold']:
        st.markdown('<h3 class="section-header">🥇 Live Gold Market Analysis</h3>', unsafe_allow_html=True)
        
        gold_data = data['gold']
        
        # Gold metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="gold-price-card">
                <h3>💰 Current Price</h3>
                <h2>${gold_data['current_price']:.2f}</h2>
                <p>per Troy Ounce</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            change_color = "#10b981" if gold_data['daily_change'] >= 0 else "#ef4444"
            change_icon = "📈" if gold_data['daily_change'] >= 0 else "📉"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Daily Change</h4>
                <h3 style="color: {change_color}">{change_icon} {gold_data['daily_change']:+.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>52W High</h4>
                <h3>${gold_data['high_52w']:.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Volatility</h4>
                <h3>{gold_data['volatility']:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Gold price chart
        gold_chart = services['visualizer'].create_gold_price_chart(gold_data)
        if gold_chart:
            st.plotly_chart(gold_chart, use_container_width=True)
    
    # Comprehensive Market Dashboard
    market_overview = services['visualizer'].create_comprehensive_market_dashboard(data)
    if market_overview:
        st.plotly_chart(market_overview, use_container_width=True)
    
    # Create tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Stocks", "₿ Crypto", "🛢️ Commodities", "💱 Forex", "🔗 Correlations"
    ])
    
    with tab1:
        if 'stocks' in data and data['stocks']:
            # Sector performance
            sector_chart = services['visualizer'].create_sector_performance_chart(data['stocks'])
            if sector_chart:
                st.plotly_chart(sector_chart, use_container_width=True)
            
            # Stock table with live data
            st.subheader("📊 Live Stock Prices")
            stock_table_data = []
            for symbol, info in data['stocks'].items():
                stock_table_data.append({
                    'Symbol': symbol,
                    'Price': f"${info['current_price']:.2f}",
                    'Change (%)': f"{info['daily_change']:+.2f}%",
                    'Volume': f"{info.get('volume', 0):,}",
                    'Market Cap': f"${info['info'].get('marketCap', 0)/1e9:.1f}B" if info['info'].get('marketCap') else 'N/A'
                })
            
            if stock_table_data:
                df = pd.DataFrame(stock_table_data)
                st.dataframe(df, use_container_width=True)
    
    with tab2:
        if 'crypto' in data and data['crypto']:
            crypto_chart = services['visualizer'].create_crypto_performance_chart(data['crypto'])
            if crypto_chart:
                st.plotly_chart(crypto_chart, use_container_width=True)
            
            # Crypto table
            st.subheader("₿ Cryptocurrency Prices")
            crypto_table_data = []
            for symbol, info in data['crypto'].items():
                crypto_table_data.append({
                    'Symbol': symbol.replace('-USD', ''),
                    'Price': f"${info['current_price']:,.2f}",
                    'Change (%)': f"{info['daily_change']:+.2f}%",
                    '24h Volume': f"{info.get('volume_24h', 0):,.0f}"
                })
            
            if crypto_table_data:
                df = pd.DataFrame(crypto_table_data)
                st.dataframe(df, use_container_width=True)
    
    with tab3:
        if 'commodities' in data and data['commodities']:
            st.subheader("🛢️ Commodities Market")
            commodities_table_data = []
            for symbol, info in data['commodities'].items():
                commodities_table_data.append({
                    'Commodity': info.get('name', symbol),
                    'Price': f"${info['current_price']:.2f}",
                    'Daily Change (%)': f"{info['daily_change']:+.2f}%"
                })
            
            if commodities_table_data:
                df = pd.DataFrame(commodities_table_data)
                st.dataframe(df, use_container_width=True)
    
    with tab4:
        if 'forex' in data and data['forex']:
            st.subheader("💱 Foreign Exchange")
            forex_table_data = []
            for symbol, info in data['forex'].items():
                forex_table_data.append({
                    'Pair': symbol.replace('=X', ''),
                    'Rate': f"{info['current_rate']:.5f}",
                    'Daily Change (%)': f"{info['daily_change']:+.2f}%"
                })
            
            if forex_table_data:
                df = pd.DataFrame(forex_table_data)
                st.dataframe(df, use_container_width=True)
    
    with tab5:
        correlation_chart = services['visualizer'].create_market_correlation_heatmap(data)
        if correlation_chart:
            st.plotly_chart(correlation_chart, use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis")

def create_enhanced_sidebar(services, data):
    """Create enhanced sidebar with dark theme"""
    with st.sidebar:
        # Language selector
        lang = st.session_state.app_settings.get('language', 'en')
        new_lang = st.selectbox(
            "🌐 Language / اللغة:",
            options=['en', 'ar'],
            index=0 if lang == 'en' else 1,
            format_func=lambda x: "🇺🇸 English" if x == 'en' else "🇸🇦 العربية"
        )
        
        if new_lang != lang:
            st.session_state.app_settings['language'] = new_lang
            st.rerun()
        
        st.divider()
        
        # Agent status with enhanced styling
        st.markdown("### 🤖 Agent Status")
        if services['ai_agent'].initialized:
            st.markdown('<p class="status-active">✅ AI Agent Active</p>', unsafe_allow_html=True)
            st.info("🧠 GPT-4 Powered")
            st.info("📊 Real-time Data")
            st.info("🌍 Global Markets")
            st.info("🥇 Live Gold Prices")
        else:
            st.markdown('<p class="status-inactive">❌ Agent Offline</p>', unsafe_allow_html=True)
            if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
                st.warning("Missing OpenAI API Key")
        
        st.divider()
        
        # Live market summary
        st.markdown("### 📈 Market Summary")
        if data:
            if 'gold' in data and data['gold']:
                gold_change = data['gold']['daily_change']
                gold_icon = "📈" if gold_change >= 0 else "📉"
                st.write(f"🥇 **Gold:** ${data['gold']['current_price']:.2f} {gold_icon} {gold_change:+.2f}%")
            
            if 'stocks' in data and data['stocks']:
                profitable_stocks = sum(1 for stock in data['stocks'].values() if stock.get('daily_change', 0) > 0)
                total_stocks = len(data['stocks'])
                st.write(f"📈 **Gainers:** {profitable_stocks}/{total_stocks}")
            
            if 'crypto' in data and data['crypto']:
                crypto_count = len(data['crypto'])
                st.write(f"₿ **Crypto:** {crypto_count} tracked")
        
        last_update = datetime.now().strftime("%H:%M:%S")
        st.write(f"🕐 **Last Update:** {last_update}")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()
        
        st.divider()
        
        # User profile
        st.markdown("### 👤 Your Profile")
        services['user_manager'].display_profile_summary()
        
        st.divider()
        
        # Enhanced conversation history
        st.markdown("### 💬 Recent Conversations")
        history = st.session_state.conversation_history[-3:]
        
        if history:
            for i, entry in enumerate(reversed(history), 1):
                with st.expander(f"💬 Q{len(history)-i+1}: {entry['question'][:25]}..."):
                    st.write(f"🕐 **Time:** {entry['timestamp']}")
                    st.write(f"❓ **Question:** {entry['question']}")
                    st.write(f"🤖 **Answer:** {entry['answer'][:150]}...")
        else:
            st.info("💭 No conversations yet. Ask me anything!")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.success("History cleared!")
            st.rerun()
        
        st.divider()
        
        # Enhanced app info
        st.markdown("### ℹ️ App Information")
        st.write(f"🔢 **Version:** {VERSION}")
        st.write(f"🎨 **Theme:** Enhanced Dark")
        st.write(f"📊 **Charts:** Interactive Plotly")
        st.write(f"🌐 **APIs:** Yahoo Finance")
        st.write(f"🤖 **AI:** OpenAI GPT-4")
        
        # Data statistics
        if data:
            total_assets = 0
            if 'stocks' in data:
                total_assets += len(data['stocks'])
            if 'crypto' in data:
                total_assets += len(data['crypto'])
            if 'commodities' in data:
                total_assets += len(data['commodities'])
            if 'forex' in data:
                total_assets += len(data['forex'])
            
            st.write(f"📈 **Total Assets:** {total_assets}")
            
            if 'csv_data' in data and not data['csv_data'].empty:
                st.write(f"📄 **CSV Records:** {len(data['csv_data'])}")

def main():
    """Enhanced main application with dark theme and comprehensive features"""
    try:
        # Initialize everything
        load_css()
        initialize_session_state()
        
        # Display enhanced header
        display_header()
        
        # Initialize services
        services = initialize_services()
        
        # Fetch comprehensive market data with caching
        @st.cache_data(ttl=300)  # 5-minute cache for real-time feel
        def get_market_data():
            return services['data_fetcher'].fetch_comprehensive_market_data()
        
        # Get live market data
        with st.spinner("📡 Fetching real-time market data from global exchanges..."):
            data = get_market_data()
        
        # Main chat interface
        create_chat_interface(services)
        
        # Enhanced dashboard with all requested features
        if data:
            create_enhanced_dashboard(services, data)
        else:
            st.error("❌ Unable to fetch market data. Please check your internet connection and try again.")
        
        # Enhanced sidebar
        create_enhanced_sidebar(services, data)
        
        # Enhanced footer with dark theme
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #b8bcc8; padding: 2rem;">
            <h4 style="color: #4A90E2; margin-bottom: 1rem;">🤖 {APP_NAME}</h4>
            <p style="margin: 0.5rem 0;">Version {VERSION} | Enhanced Dark Theme | Real-time Global Markets</p>
            <p style="margin: 0.5rem 0;">🥇 Live Gold Prices | ₿ Cryptocurrency | 📈 Stocks | 💱 Forex | 🛢️ Commodities</p>
            <p style="margin: 0.5rem 0; font-size: 0.9em;">⚠️ For educational purposes only. Always consult qualified financial advisors for investment decisions.</p>
            <p style="margin-top: 1rem; font-size: 0.8em; color: #6c7293;">
                Powered by OpenAI GPT-4 | Yahoo Finance API | Plotly Visualizations
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"💥 Application error: {e}")
        if DEBUG:
            st.exception(e)

if __name__ == "__main__":
    main()