"""
Configuration package for Expert Financial Risk Analysis AI.

This package contains all configuration settings, environment variables,
and application constants used throughout the financial AI platform.

Modules:
    settings: Main configuration file with API keys, symbols, and app settings
"""

try:
    from .settings import *
    
    # Make key configuration variables easily accessible
    __all__ = [
        # App Configuration
        'APP_NAME',
        'VERSION', 
        'DEBUG',
        'DEFAULT_LANGUAGE',
        'SUPPORTED_LANGUAGES',
        
        # API Keys
        'OPENAI_API_KEY',
        'ALPHA_API_KEY',
        'FMP_API_KEY',
        'OER_API_KEY',
        'GOLDAP_API_KEY',
        'FINNHUB_API_KEY',
        'NINJAS_API_KEY',
        'MARKETSTACK_API_KEY',
        'POLYGON_API_KEY',
        'NEWS_API_KEY',
        'FRED_API_KEY',
        'QUANDL_API_KEY',
        'SERPAPI_KEY',
        
        # Financial Configuration
        'RISK_FREE_RATE',
        'MARKET_RETURN',
        'COMPANY_SYMBOLS',
        'CRYPTO_SYMBOLS',
        'FOREX_PAIRS',
        'COMMODITIES',
        'ECONOMIC_INDICATORS',
        
        # Cache Configuration
        'CACHE_TTL_SHORT',
        'CACHE_TTL_MEDIUM', 
        'CACHE_TTL_LONG',
        'CACHE_TTL_VERY_LONG',
        
        # Data Processing
        'CSV_DATA_PATH',
        'SPLIT_CSV_FOLDER',
        'MAX_CSV_ROWS_FOR_PROCESSING',
        'CSV_CHUNK_SIZE',
        
        # Risk Analysis
        'RISK_METRICS',
        'RISK_CATEGORIES',
        
        # AI Configuration
        'AI_MODEL_CONFIG',
        
        # Other configurations
        'WB_COUNTRIES',
        'SEARCH_CONFIG',
        'DATA_SOURCE_PRIORITY',
        'PORTFOLIO_CONFIG',
        'BACKTEST_CONFIG',
        'REFRESH_INTERVALS',
        'LANGUAGE_CONFIG',
        'ERROR_HANDLING',
        'FEATURES'
    ]
    
except ImportError as e:
    # Graceful fallback if settings cannot be imported
    print(f"Warning: Could not import configuration settings - {e}")
    print("Please ensure config/settings.py exists and is properly configured.")
    
    # Provide minimal fallback configuration
    APP_NAME = "Expert Financial Risk Analysis AI"
    VERSION = "2.0.0"
    DEBUG = False
    DEFAULT_LANGUAGE = "en"
    
    __all__ = ['APP_NAME', 'VERSION', 'DEBUG', 'DEFAULT_LANGUAGE']

# Package metadata
__version__ = "2.0.0"
__author__ = "Financial AI Team"
__description__ = "Configuration package for Financial AI Assistant"
__package_name__ = "config"