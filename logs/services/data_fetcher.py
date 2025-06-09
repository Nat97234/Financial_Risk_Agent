import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import wbgapi as wb
import numpy as np
import os
import glob
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataFetcher:
    def __init__(self):
        self.company_symbols = COMPANY_SYMBOLS
        self.crypto_symbols = CRYPTO_SYMBOLS
        self.forex_pairs = FOREX_PAIRS
        self.commodities = COMMODITIES
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FinancialAI/2.0 (Professional Financial Analysis Tool)'
        })
    
    def _make_request(self, url: str, params: Dict = None, timeout: int = 30) -> Optional[Dict]:
        """Enhanced request handler with retry logic"""
        for attempt in range(ERROR_HANDLING["MAX_RETRIES"]):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < ERROR_HANDLING["MAX_RETRIES"] - 1:
                    time.sleep(ERROR_HANDLING["RETRY_DELAY"])
                else:
                    logger.error(f"All retry attempts failed for URL: {url}")
        return None

    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_enhanced_yfinance_data(_self, symbols: List[str] = None, period: str = "1y") -> Dict[str, Any]:
        """Fetch comprehensive Yahoo Finance data with technical indicators"""
        symbols = symbols or _self.company_symbols
        all_data = {}
        
        def fetch_single_symbol(symbol: str) -> Tuple[str, Dict]:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                hist = ticker.history(period=period)
                
                # Get company info
                info = ticker.info
                
                # Calculate technical indicators
                technical_data = {}
                if not hist.empty and len(hist) > 50:
                    technical_data = _self._calculate_technical_indicators(hist)
                
                # Get financial statements
                financials = {}
                try:
                    financials = {
                        'income_stmt': ticker.income_stmt,
                        'balance_sheet': ticker.balance_sheet,
                        'cash_flow': ticker.cashflow
                    }
                except:
                    pass
                
                return symbol, {
                    'history': hist,
                    'info': info,
                    'technical': technical_data,
                    'financials': financials,
                    'last_updated': datetime.now()
                }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, None
        
        # Use ThreadPoolExecutor for parallel data fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_single_symbol, symbol): symbol 
                              for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data:
                    all_data[symbol] = data
        
        return all_data

    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        if hist.empty or len(hist) < 20:
            return {}
        
        indicators = {}
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        
        try:
            # Moving Averages
            indicators['sma_20'] = close.rolling(window=20).mean()
            indicators['sma_50'] = close.rolling(window=50).mean()
            indicators['sma_200'] = close.rolling(window=200).mean()
            indicators['ema_12'] = close.ewm(span=12).mean()
            indicators['ema_26'] = close.ewm(span=26).mean()
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators['bb_upper'] = sma_20 + (std_20 * 2)
            indicators['bb_lower'] = sma_20 - (std_20 * 2)
            indicators['bb_middle'] = sma_20
            
            # Volatility
            indicators['volatility_30d'] = close.pct_change().rolling(window=30).std() * np.sqrt(252)
            indicators['volatility_90d'] = close.pct_change().rolling(window=90).std() * np.sqrt(252)
            
            # Volume indicators
            indicators['volume_sma'] = volume.rolling(window=20).mean()
            indicators['volume_ratio'] = volume / indicators['volume_sma']
            
            # Support and Resistance levels
            indicators['resistance'] = high.rolling(window=20).max()
            indicators['support'] = low.rolling(window=20).min()
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators

    @st.cache_data(ttl=CACHE_TTL_LONG)
    def get_enhanced_world_bank_data(_self) -> Dict[str, Any]:
        """Fetch comprehensive World Bank economic data"""
        try:
            indicators = {
                'gdp': 'NY.GDP.MKTP.CD',
                'gdp_per_capita': 'NY.GDP.PCAP.CD',
                'inflation': 'FP.CPI.TOTL.ZG',
                'unemployment': 'SL.UEM.TOTL.ZS',
                'population': 'SP.POP.TOTL',
                'trade_balance': 'NE.RSB.GNFS.ZS',
                'government_debt': 'GC.DOD.TOTL.GD.ZS',
                'interest_rates': 'FR.INR.RINR',
                'exchange_rates': 'PA.NUS.FCRF',
                'fdi_inflows': 'BX.KLT.DINV.WD.GD.ZS'
            }
            
            wb_data = {}
            current_year = datetime.now().year
            years = range(current_year - 10, current_year + 1)
            
            for indicator_name, indicator_code in indicators.items():
                try:
                    data = wb.data.DataFrame(
                        indicator_code, 
                        WB_COUNTRIES, 
                        time=years,
                        skipBlanks=True
                    )
                    wb_data[indicator_name] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch {indicator_name}: {e}")
                    continue
            
            return wb_data
        except Exception as e:
            logger.error(f"Error fetching World Bank data: {e}")
            return {}

    @st.cache_data(ttl=CACHE_TTL_MEDIUM)
    def get_financial_modeling_prep_data(_self, symbols: List[str] = None) -> Dict[str, Any]:
        """Fetch data from Financial Modeling Prep API"""
        if not FMP_API_KEY:
            return {}
        
        symbols = symbols or _self.company_symbols[:10]  # Limit for API quota
        fmp_data = {}
        
        for symbol in symbols:
            try:
                # Company profile
                profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
                profile_data = _self._make_request(profile_url, {'apikey': FMP_API_KEY})
                
                # Financial ratios
                ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
                ratios_data = _self._make_request(ratios_url, {'apikey': FMP_API_KEY})
                
                # DCF valuation
                dcf_url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{symbol}"
                dcf_data = _self._make_request(dcf_url, {'apikey': FMP_API_KEY})
                
                # Stock news
                news_url = f"https://financialmodelingprep.com/api/v3/stock_news"
                news_data = _self._make_request(news_url, {
                    'apikey': FMP_API_KEY,
                    'tickers': symbol,
                    'limit': 5
                })
                
                if any([profile_data, ratios_data, dcf_data, news_data]):
                    fmp_data[symbol] = {
                        'profile': profile_data[0] if profile_data else None,
                        'ratios': ratios_data[:5] if ratios_data else None,
                        'dcf': dcf_data[0] if dcf_data else None,
                        'news': news_data[:5] if news_data else None
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching FMP data for {symbol}: {e}")
                continue
        
        return fmp_data

    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_alpha_vantage_enhanced(_self, symbols: List[str] = None) -> Dict[str, Any]:
        """Enhanced Alpha Vantage data with fundamental analysis"""
        if not ALPHA_API_KEY:
            return {}
        
        symbols = symbols or _self.company_symbols[:5]  # API limit consideration
        av_data = {}
        
        for symbol in symbols:
            try:
                # Time series data
                ts_url = "https://www.alphavantage.co/query"
                ts_params = {
                    'function': 'TIME_SERIES_DAILY_ADJUSTED',
                    'symbol': symbol,
                    'apikey': ALPHA_API_KEY,
                    'outputsize': 'compact'
                }
                ts_data = _self._make_request(ts_url, ts_params)
                
                # Company overview
                overview_params = {
                    'function': 'OVERVIEW',
                    'symbol': symbol,
                    'apikey': ALPHA_API_KEY
                }
                overview_data = _self._make_request(ts_url, overview_params)
                
                # Earnings data
                earnings_params = {
                    'function': 'EARNINGS',
                    'symbol': symbol,
                    'apikey': ALPHA_API_KEY
                }
                earnings_data = _self._make_request(ts_url, earnings_params)
                
                if ts_data or overview_data or earnings_data:
                    av_data[symbol] = {
                        'time_series': ts_data,
                        'overview': overview_data,
                        'earnings': earnings_data,
                        'last_updated': datetime.now()
                    }
                
                # Rate limiting
                time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
                
            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
                continue
                
        return av_data

    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_crypto_data(_self) -> Dict[str, Any]:
        """Fetch cryptocurrency data"""
        crypto_data = {}
        
        try:
            # Fetch crypto data using yfinance
            for symbol in _self.crypto_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")
                    info = ticker.info
                    
                    if not hist.empty:
                        # Calculate crypto-specific metrics
                        returns = hist['Close'].pct_change()
                        volatility = returns.std() * np.sqrt(365)  # Annual volatility
                        
                        crypto_data[symbol] = {
                            'history': hist,
                            'info': info,
                            'volatility': volatility,
                            'returns': returns,
                            'current_price': hist['Close'].iloc[-1],
                            'volume_24h': hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                        }
                except Exception as e:
                    logger.error(f"Error fetching crypto data for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in crypto data fetching: {e}")
        
        return crypto_data

    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_forex_data(_self) -> Dict[str, Any]:
        """Fetch forex data"""
        forex_data = {}
        
        try:
            for pair in _self.forex_pairs:
                try:
                    ticker = yf.Ticker(pair)
                    hist = ticker.history(period="3mo")
                    
                    if not hist.empty:
                        # Calculate forex-specific metrics
                        returns = hist['Close'].pct_change()
                        volatility = returns.std() * np.sqrt(252)
                        
                        forex_data[pair] = {
                            'history': hist,
                            'volatility': volatility,
                            'returns': returns,
                            'current_rate': hist['Close'].iloc[-1],
                            'change_1d': returns.iloc[-1] * 100
                        }
                except Exception as e:
                    logger.error(f"Error fetching forex data for {pair}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in forex data fetching: {e}")
        
        return forex_data

    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_commodities_data(_self) -> Dict[str, Any]:
        """Fetch commodities data"""
        commodities_data = {}
        
        try:
            for commodity in _self.commodities:
                try:
                    ticker = yf.Ticker(commodity)
                    hist = ticker.history(period="6mo")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change()
                        volatility = returns.std() * np.sqrt(252)
                        
                        commodities_data[commodity] = {
                            'history': hist,
                            'volatility': volatility,
                            'returns': returns,
                            'current_price': hist['Close'].iloc[-1],
                            'change_ytd': ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                        }
                except Exception as e:
                    logger.error(f"Error fetching commodity data for {commodity}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in commodities data fetching: {e}")
        
        return commodities_data

    @st.cache_data(ttl=CACHE_TTL_MEDIUM)
    def get_economic_indicators(_self) -> Dict[str, Any]:
        """Fetch real-time economic indicators"""
        indicators = {}
        
        try:
            # VIX (Fear & Greed Index)
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1mo")
            if not vix_data.empty:
                indicators['VIX'] = {
                    'current': vix_data['Close'].iloc[-1],
                    'history': vix_data,
                    'interpretation': _self._interpret_vix(vix_data['Close'].iloc[-1])
                }
            
            # DXY (Dollar Index)
            dxy_ticker = yf.Ticker("DX-Y.NYB")
            dxy_data = dxy_ticker.history(period="1mo")
            if not dxy_data.empty:
                indicators['DXY'] = {
                    'current': dxy_data['Close'].iloc[-1],
                    'history': dxy_data
                }
            
            # 10-Year Treasury Yield
            tnx_ticker = yf.Ticker("^TNX")
            tnx_data = tnx_ticker.history(period="1mo")
            if not tnx_data.empty:
                indicators['TNX'] = {
                    'current': tnx_data['Close'].iloc[-1],
                    'history': tnx_data
                }
                
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
        
        return indicators

    def _interpret_vix(self, vix_value: float) -> str:
        """Interpret VIX levels"""
        if vix_value < 15:
            return "Low volatility - Market complacency"
        elif vix_value < 25:
            return "Normal volatility - Stable market conditions"
        elif vix_value < 35:
            return "High volatility - Market uncertainty"
        else:
            return "Extreme volatility - Market panic/crisis"

    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_gold_prices_enhanced(_self) -> Dict[str, Any]:
        """Enhanced gold price data with analysis"""
        gold_data = {}
        
        try:
            # Multiple gold sources
            gold_symbols = ["GC=F", "GOLD", "GLD"]  # Gold futures, Barrick Gold, Gold ETF
            
            for symbol in gold_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="6mo")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        
                        # Calculate gold-specific metrics
                        returns = hist['Close'].pct_change()
                        volatility = returns.std() * np.sqrt(252)
                        
                        # Gold inflation hedge analysis
                        correlation_with_inflation = _self._calculate_inflation_hedge_score(hist)
                        
                        gold_data[symbol] = {
                            'current_price': current_price,
                            'history': hist,
                            'volatility': volatility,
                            'returns': returns,
                            'inflation_hedge_score': correlation_with_inflation,
                            'trend': _self._determine_trend(hist['Close'])
                        }
                        
                except Exception as e:
                    logger.error(f"Error fetching gold data for {symbol}: {e}")
                    continue
            
            # Convert to standard units if we have futures data
            if "GC=F" in gold_data:
                price_per_ounce = gold_data["GC=F"]["current_price"]
                gold_data["prices"] = {
                    'ounce': price_per_ounce,
                    'gram': price_per_ounce / 31.1035,
                    'kilogram': price_per_ounce * 32.1507,
                    'pound': price_per_ounce * 14.5833
                }
                
        except Exception as e:
            logger.error(f"Error in enhanced gold data fetching: {e}")
        
        return gold_data

    def _calculate_inflation_hedge_score(self, price_history: pd.DataFrame) -> float:
        """Calculate how well an asset hedges against inflation"""
        try:
            # Simplified inflation hedge calculation
            # In practice, you'd compare with actual inflation data
            returns = price_history['Close'].pct_change()
            volatility = returns.std()
            avg_return = returns.mean()
            
            # Higher positive returns with lower volatility = better hedge
            if volatility > 0:
                hedge_score = (avg_return / volatility) * 100
                return max(-100, min(100, hedge_score))  # Normalize to -100 to 100
            return 0
        except:
            return 0

    def _determine_trend(self, prices: pd.Series) -> str:
        """Determine price trend"""
        if len(prices) < 10:
            return "Insufficient data"
        
        recent_avg = prices.tail(5).mean()
        older_avg = prices.head(5).mean()
        
        change_pct = ((recent_avg - older_avg) / older_avg) * 100
        
        if change_pct > 5:
            return "Strong Uptrend"
        elif change_pct > 1:
            return "Uptrend"
        elif change_pct < -5:
            return "Strong Downtrend"
        elif change_pct < -1:
            return "Downtrend"
        else:
            return "Sideways"

    def load_enhanced_csv_data(self) -> pd.DataFrame:
        """Load CSV data with enhanced processing"""
        try:
            # Try split files first
            if os.path.exists(SPLIT_CSV_FOLDER):
                csv_files = glob.glob(f'{SPLIT_CSV_FOLDER}/*.csv')
                if csv_files:
                    dataframes = []
                    for file in sorted(csv_files):
                        try:
                            df = pd.read_csv(file)
                            dataframes.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to load {file}: {e}")
                            continue
                    
                    if dataframes:
                        combined_df = pd.concat(dataframes, ignore_index=True)
                        # Enhanced data cleaning
                        combined_df = self._clean_financial_data(combined_df)
                        return combined_df
            
            # Fallback to single file
            if os.path.exists(CSV_DATA_PATH):
                df = pd.read_csv(CSV_DATA_PATH)
                return self._clean_financial_data(df)
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
        
        return pd.DataFrame()

    def _clean_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning for financial data"""
        if df.empty:
            return df
        
        try:
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Handle text columns  
            text_columns = df.select_dtypes(include=['object']).columns
            df[text_columns] = df[text_columns].fillna('Unknown')
            
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            # Handle date columns if they exist
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    continue
            
            # Handle financial ratios (remove extreme outliers)
            ratio_columns = [col for col in df.columns if any(term in col.lower() 
                           for term in ['ratio', 'margin', 'return', 'yield'])]
            for col in ratio_columns:
                if col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error cleaning financial data: {e}")
        
        return df

    def fetch_comprehensive_data(self) -> Dict[str, Any]:
        """Fetch all available financial data"""
        logger.info("Starting comprehensive data fetch...")
        
        data = {
            "timestamp": datetime.now(),
            "csv_data": self.load_enhanced_csv_data(),
            "stocks": self.get_enhanced_yfinance_data(),
            "world_bank": self.get_enhanced_world_bank_data(),
            "alpha_vantage": self.get_alpha_vantage_enhanced(),
            "fmp": self.get_financial_modeling_prep_data(),
            "crypto": self.get_crypto_data(),
            "forex": self.get_forex_data(),
            "commodities": self.get_commodities_data(),
            "economic_indicators": self.get_economic_indicators(),
            "gold": self.get_gold_prices_enhanced()
        }
        
        # Calculate data quality score
        data["data_quality"] = self._calculate_data_quality(data)
        
        logger.info("Comprehensive data fetch completed")
        return data

    def _calculate_data_quality(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data quality metrics"""
        quality_scores = {}
        
        try:
            # CSV data quality
            csv_data = data.get("csv_data", pd.DataFrame())
            if not csv_data.empty:
                completeness = 1 - (csv_data.isnull().sum().sum() / (len(csv_data) * len(csv_data.columns)))
                quality_scores["csv"] = max(0, min(1, completeness))
            
            # Stocks data quality
            stocks_data = data.get("stocks", {})
            if stocks_data:
                successful_fetches = sum(1 for v in stocks_data.values() if v and 'history' in v)
                quality_scores["stocks"] = successful_fetches / len(self.company_symbols)
            
            # Overall quality score
            if quality_scores:
                quality_scores["overall"] = sum(quality_scores.values()) / len(quality_scores)
            else:
                quality_scores["overall"] = 0
                
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            quality_scores["overall"] = 0
        
        return quality_scores

    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and trading hours"""
        try:
            import pytz
            from datetime import datetime
            
            # Major market timezones
            markets = {
                'NYSE': 'America/New_York',
                'NASDAQ': 'America/New_York', 
                'LSE': 'Europe/London',
                'TSE': 'Asia/Tokyo',
                'SSE': 'Asia/Shanghai',
                'BSE': 'Asia/Kolkata'
            }
            
            market_status = {}
            
            for market, timezone in markets.items():
                try:
                    tz = pytz.timezone(timezone)
                    local_time = datetime.now(tz)
                    
                    # Simplified market hours (you can enhance this)
                    is_weekday = local_time.weekday() < 5
                    hour = local_time.hour
                    
                    if market in ['NYSE', 'NASDAQ']:
                        is_open = is_weekday and 9 <= hour < 16
                    elif market == 'LSE':
                        is_open = is_weekday and 8 <= hour < 16
                    elif market == 'TSE':
                        is_open = is_weekday and (9 <= hour < 11 or 12 <= hour < 15)
                    else:
                        is_open = is_weekday and 9 <= hour < 15
                    
                    market_status[market] = {
                        'is_open': is_open,
                        'local_time': local_time.strftime('%H:%M %Z'),
                        'status': 'Open' if is_open else 'Closed'
                    }
                    
                except Exception as e:
                    logger.error(f"Error getting status for {market}: {e}")
                    market_status[market] = {'status': 'Unknown'}
            
            return market_status
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {}

    def get_data_freshness(self) -> Dict[str, str]:
        """Check how fresh the cached data is"""
        freshness = {}
        
        # This is a simplified version - you can enhance based on your caching strategy
        try:
            current_time = datetime.now()
            
            # Check if it's market hours for real-time data importance
            market_status = self.get_market_status()
            any_market_open = any(status.get('is_open', False) for status in market_status.values())
            
            if any_market_open:
                freshness['recommendation'] = 'Real-time data recommended - markets are open'
                freshness['priority'] = 'high'
            else:
                freshness['recommendation'] = 'Cached data acceptable - markets are closed'
                freshness['priority'] = 'low'
            
            freshness['last_check'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            freshness['status'] = 'Unknown'
        
        return freshness