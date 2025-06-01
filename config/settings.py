import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys - Updated with current APIs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")  # Financial Modeling Prep
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")
GOLDAP_API_KEY = os.getenv("GOLDAP_API")
FINNHUB_API_KEY = os.getenv("FINNHUB_API")
NINJAS_API_KEY = os.getenv("NINJAS_API")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # For financial news
FRED_API_KEY = os.getenv("FRED_API_KEY")  # Federal Reserve Economic Data
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY")  # Economic data
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # For enhanced web search

# Application Configuration
APP_NAME = "Expert Financial Risk Analysis AI"
VERSION = "2.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Multilingual Support
SUPPORTED_LANGUAGES = ["en", "ar"]
DEFAULT_LANGUAGE = "en"

# Financial Analysis Configuration
RISK_FREE_RATE = 0.02  # 2% risk-free rate (updated regularly)
MARKET_RETURN = 0.10   # 10% market return assumption

# Enhanced Company Symbols - Major global companies
COMPANY_SYMBOLS = [
    # US Tech Giants
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    # Financial Services
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    # Healthcare & Pharma
    "JNJ", "PFE", "UNH", "ABBV", "MRK",
    # Consumer & Retail
    "WMT", "PG", "KO", "PEP", "HD",
    # Energy & Commodities
    "XOM", "CVX", "COP", "SLB",
    # International
    "TSM", "ASML", "NVO", "TM", "BABA",
    # ETFs for diversification analysis
    "SPY", "QQQ", "VTI", "BND", "GLD"
]

# Cryptocurrency symbols for digital asset analysis
CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"
]

# Forex pairs for currency analysis
FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X"
]

# Commodities for inflation and economic analysis
COMMODITIES = [
    "GC=F",  # Gold
    "SI=F",  # Silver
    "CL=F",  # Crude Oil
    "NG=F",  # Natural Gas
    "ZC=F",  # Corn
    "ZW=F",  # Wheat
    "ZS=F"   # Soybeans
]

# Economic Indicators
ECONOMIC_INDICATORS = {
    "US": ["GDP", "CPI", "UNEMPLOYMENT", "INTEREST_RATES", "RETAIL_SALES"],
    "EU": ["GDP", "CPI", "UNEMPLOYMENT", "ECB_RATES"],
    "CHINA": ["GDP", "CPI", "MANUFACTURING_PMI"],
    "GLOBAL": ["OIL_PRICES", "GOLD_PRICES", "VIX", "DXY"]
}

# CSV Data Paths
CSV_DATA_PATH = 'financial_risk_analysis_large.csv'
SPLIT_CSV_FOLDER = 'split_files'

# Cache Configuration - Enhanced for better performance
CACHE_TTL_SHORT = 300      # 5 minutes for real-time data
CACHE_TTL_MEDIUM = 1800    # 30 minutes for market data
CACHE_TTL_LONG = 3600      # 1 hour for historical data
CACHE_TTL_VERY_LONG = 86400 # 24 hours for fundamental data

# Enhanced World Bank Countries
WB_COUNTRIES = [
    'USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA', 'ITA', 'CAN',
    'RUS', 'KOR', 'ESP', 'AUS', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'TWN'
]

# CSV Processing Configuration
MAX_CSV_ROWS_FOR_PROCESSING = 50000  # Increased for better analysis
CSV_CHUNK_SIZE = 2000

# Risk Analysis Configuration
RISK_METRICS = {
    "VaR_CONFIDENCE_LEVELS": [0.95, 0.99],  # Value at Risk confidence levels
    "STRESS_TEST_SCENARIOS": ["2008_CRISIS", "COVID_2020", "DOT_COM_BUBBLE"],
    "CORRELATION_THRESHOLD": 0.7,  # High correlation threshold
    "VOLATILITY_PERIODS": [30, 60, 90, 252]  # Days for volatility calculations
}

# AI Agent Configuration
AI_MODEL_CONFIG = {
    "PRIMARY_MODEL": "gpt-4-turbo-preview",  # Latest GPT-4 model
    "FALLBACK_MODEL": "gpt-3.5-turbo",
    "TEMPERATURE": 0.1,  # Low temperature for financial accuracy
    "MAX_TOKENS": 4000,
    "CONTEXT_WINDOW": 16000
}

# Web Search Configuration
SEARCH_CONFIG = {
    "MAX_RESULTS": 10,
    "SEARCH_DOMAINS": [
        "bloomberg.com", "reuters.com", "wsj.com", "ft.com",
        "marketwatch.com", "cnbc.com", "yahoo.com", "investing.com"
    ],
    "EXCLUDED_DOMAINS": ["reddit.com", "twitter.com"],  # Exclude social media
    "SEARCH_LANGUAGES": ["en", "ar"]
}

# Financial Data Sources Priority
DATA_SOURCE_PRIORITY = [
    "bloomberg_api",
    "alpha_vantage",
    "financial_modeling_prep",
    "yahoo_finance",
    "polygon",
    "finnhub"
]

# Risk Assessment Categories
RISK_CATEGORIES = {
    "CONSERVATIVE": {"min_score": 1, "max_score": 3, "description": "Low risk tolerance"},
    "MODERATE": {"min_score": 4, "max_score": 6, "description": "Balanced risk approach"},
    "AGGRESSIVE": {"min_score": 7, "max_score": 8, "description": "High risk tolerance"},
    "SPECULATIVE": {"min_score": 9, "max_score": 10, "description": "Very high risk tolerance"}
}

# Portfolio Optimization Parameters
PORTFOLIO_CONFIG = {
    "MIN_WEIGHT": 0.01,  # Minimum 1% allocation
    "MAX_WEIGHT": 0.40,  # Maximum 40% allocation
    "REBALANCING_THRESHOLD": 0.05,  # 5% deviation triggers rebalancing
    "OPTIMIZATION_METHODS": ["mean_variance", "risk_parity", "black_litterman"]
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    "DEFAULT_PERIOD": "5Y",
    "BENCHMARK": "SPY",
    "TRANSACTION_COST": 0.001,  # 0.1% transaction cost
    "STARTING_CAPITAL": 100000  # $100,000 default portfolio
}

# Real-time Data Refresh Intervals (seconds)
REFRESH_INTERVALS = {
    "MARKET_DATA": 60,      # 1 minute for market data
    "NEWS": 300,            # 5 minutes for news
    "ECONOMIC_DATA": 1800,  # 30 minutes for economic indicators
    "FUNDAMENTAL_DATA": 86400  # 24 hours for fundamental data
}

# Language-specific configurations
LANGUAGE_CONFIG = {
    "ar": {
        "currency_symbol": "ر.س",  # Saudi Riyal symbol
        "date_format": "%d/%m/%Y",
        "number_format": "arabic",
        "rtl": True
    },
    "en": {
        "currency_symbol": "$",
        "date_format": "%m/%d/%Y", 
        "number_format": "western",
        "rtl": False
    }
}

# Enhanced Error Handling
ERROR_HANDLING = {
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 1,  # seconds
    "TIMEOUT": 30,     # seconds
    "LOG_LEVEL": "INFO"
}

# Feature Flags
FEATURES = {
    "ENABLE_CRYPTO_ANALYSIS": True,
    "ENABLE_FOREX_ANALYSIS": True,
    "ENABLE_COMMODITIES": True,
    "ENABLE_OPTIONS_ANALYSIS": True,
    "ENABLE_ESG_SCORING": True,
    "ENABLE_SENTIMENT_ANALYSIS": True,
    "ENABLE_TECHNICAL_INDICATORS": True,
    "ENABLE_FUNDAMENTAL_ANALYSIS": True
}