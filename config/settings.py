"""
Configuration settings for Expert Financial Risk Analysis AI
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Application Configuration
APP_NAME = "Expert Financial Risk Analysis AI"
VERSION = "2.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Supported Languages
SUPPORTED_LANGUAGES = ["en", "ar"]
DEFAULT_LANGUAGE = "en"

# Financial Configuration
RISK_FREE_RATE = 0.02
MARKET_RETURN = 0.10

# Company Symbols
COMPANY_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    "JNJ", "PFE", "UNH", "ABBV", "MRK"
]

# Cache Configuration
CACHE_TTL_SHORT = 300
CACHE_TTL_MEDIUM = 1800
CACHE_TTL_LONG = 3600

# AI Model Configuration
AI_MODEL_CONFIG = {
    "PRIMARY_MODEL": "gpt-4",
    "TEMPERATURE": 0.1,
    "MAX_TOKENS": 4000
}

# CSV Configuration
CSV_DATA_PATH = 'financial_risk_analysis_large.csv'
MAX_CSV_ROWS_FOR_PROCESSING = 10000
CSV_CHUNK_SIZE = 1000
