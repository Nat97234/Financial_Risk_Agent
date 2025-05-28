import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")
GOLDAP_API_KEY = os.getenv("GOLDAP_API")
FINNHUB_API_KEY = os.getenv("FINNHUB_API")
NINJAS_API_KEY = os.getenv("NINJAS_API")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Application Configuration
APP_NAME = "Financial AI Assistant"
VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Data Configuration
COMPANY_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
    "GOOGL", "META", "BRK-B", "AVGO", "TSM",
    "TM", "BABA", "V", "WMT", "JPM", "NFLX",
    "AMD", "CRM", "ORCL", "ADBE"
]

# CSV Data Paths
CSV_DATA_PATH = 'financial_risk_analysis_large.csv'  # Fallback single file
SPLIT_CSV_FOLDER = 'split_files'  # Main split files folder

# Cache Configuration
CACHE_TTL_SHORT = 300  # 5 minutes
CACHE_TTL_LONG = 3600  # 1 hour

# World Bank Countries
WB_COUNTRIES = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA']

# CSV Processing Configuration
MAX_CSV_ROWS_FOR_PROCESSING = 10000  # Limit rows for document processing to avoid memory issues
CSV_CHUNK_SIZE = 1000  # Process CSV in chunks for better memory management