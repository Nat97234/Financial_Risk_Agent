import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import wbgapi as wb
import os
import glob
from config.settings import *

class DataFetcher:
    def __init__(self):
        self.company_symbols = COMPANY_SYMBOLS
    
    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_yfinance_data(_self, symbols=None, period="6mo"):
        """Fetch Yahoo Finance data silently"""
        symbols = symbols or _self.company_symbols
        all_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if not hist.empty:
                    all_data[symbol] = {
                        'history': hist,
                        'info': info
                    }
            except Exception as e:
                # Silent error handling - no messages shown to user
                continue
        
        return all_data
    
    @st.cache_data(ttl=CACHE_TTL_LONG)
    def get_world_bank_data(_self):
        """Fetch World Bank data silently"""
        try:
            gdp_data = wb.data.DataFrame('NY.GDP.MKTP.CD', WB_COUNTRIES, time=range(2018, 2023))
            inflation_data = wb.data.DataFrame('FP.CPI.TOTL.ZG', WB_COUNTRIES, time=range(2018, 2023))
            unemployment_data = wb.data.DataFrame('SL.UEM.TOTL.ZS', WB_COUNTRIES, time=range(2018, 2023))
            
            return {
                'gdp': gdp_data,
                'inflation': inflation_data,
                'unemployment': unemployment_data
            }
        except Exception as e:
            # Silent error handling
            return {}
    
    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_stock_data_alpha(_self, symbols=None):
        """Fetch Alpha Vantage stock data with yfinance fallback - silently"""
        symbols = symbols or _self.company_symbols
        all_data = {}
        
        for symbol in symbols:
            if ALPHA_API_KEY:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_API_KEY}&outputsize=compact"
                try:
                    response = requests.get(url)
                    data = response.json()
                    if 'Time Series (Daily)' in data:
                        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float)
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        all_data[symbol] = df
                        continue
                except Exception as e:
                    # Silent error - try fallback
                    pass
            
            # Fallback to yfinance
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
                if not hist.empty:
                    all_data[symbol] = hist
            except Exception as e:
                # Silent error handling
                continue
        
        return all_data
    
    @st.cache_data(ttl=CACHE_TTL_SHORT)
    def get_gold_prices(_self):
        """Fetch gold prices from multiple sources - silently"""
        # Try GoldAPI first
        if GOLDAP_API_KEY:
            url = f"https://www.goldapi.io/api/XAU/USD?apikey={GOLDAP_API_KEY}"
            try:
                response = requests.get(url)
                data = response.json()
                if 'price' in data:
                    return {
                        'ounce': data['price'],
                        'gram': data['price'] / 31.1035,
                        'kilogram': data['price'] * 32.1507
                    }
            except Exception as e:
                # Silent error - try fallback
                pass
        
        # Fallback to yfinance for gold futures
        try:
            gold_ticker = yf.Ticker("GC=F")
            hist = gold_ticker.history(period="1d")
            if not hist.empty:
                price_per_ounce = hist['Close'].iloc[-1]
                return {
                    'ounce': price_per_ounce,
                    'gram': price_per_ounce / 31.1035,
                    'kilogram': price_per_ounce * 32.1507
                }
        except Exception as e:
            # Silent error handling
            pass
        
        # Return zero prices if all methods fail
        return {'ounce': 0, 'gram': 0, 'kilogram': 0}
    
    def load_split_csv_data(self):
        """Load and combine all CSV files from split_files folder - COMPLETELY SILENT"""
        try:
            # Check if split_files folder exists
            if not os.path.exists('split_files'):
                return pd.DataFrame()
            
            # Get all CSV files in the split_files folder
            csv_files = glob.glob('split_files/*.csv')
            
            if not csv_files:
                return pd.DataFrame()
            
            # Sort files to ensure consistent order
            csv_files.sort()
            
            # Load and combine all CSV files - NO PROGRESS MESSAGES OR STATUS UPDATES
            combined_data = []
            
            for file_path in csv_files:
                try:
                    # Load individual CSV file SILENTLY
                    df = pd.read_csv(file_path)
                    combined_data.append(df)
                    # NO SUCCESS MESSAGES - completely silent
                except Exception as e:
                    # NO ERROR MESSAGES - skip failed files silently
                    continue
            
            if combined_data:
                # Combine all dataframes
                final_df = pd.concat(combined_data, ignore_index=True)
                
                # Remove duplicate rows silently
                final_df = final_df.drop_duplicates()
                
                # NO SUCCESS MESSAGES about combination
                # NO DATA SUMMARY DISPLAY
                
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            # Silent error handling - no messages to user
            return pd.DataFrame()
    
    def load_csv_data(self):
        """Load CSV data - try split files first, then fallback to single file - SILENT"""
        # First try to load from split files (silently)
        split_data = self.load_split_csv_data()
        if not split_data.empty:
            return split_data
        
        # Fallback to original single file approach
        try:
            return pd.read_csv(CSV_DATA_PATH)
        except FileNotFoundError:
            # NO WARNING MESSAGES - silent fallback
            return pd.DataFrame()
    
    def fetch_all_data(self):
        """Fetch all data from different sources - COMPLETELY SILENT VERSION"""
        return {
            "csv_data": self.load_csv_data(),
            "yfinance_data": self.get_yfinance_data(),
            "world_bank_data": self.get_world_bank_data(),
            "alpha_data": self.get_stock_data_alpha(),
            "gold_prices": self.get_gold_prices()
        }
    
    # Optional: Debug method for developers (only shows if explicitly called)
    def get_data_status(self):
        """Get data loading status - only for debugging purposes"""
        status = {}
        
        # Check CSV data
        csv_data = self.load_csv_data()
        status['csv'] = {
            'loaded': not csv_data.empty,
            'rows': len(csv_data) if not csv_data.empty else 0,
            'columns': len(csv_data.columns) if not csv_data.empty else 0
        }
        
        # Check other data sources
        yfinance_data = self.get_yfinance_data()
        status['yfinance'] = {
            'loaded': len(yfinance_data) > 0,
            'symbols': len(yfinance_data)
        }
        
        world_bank_data = self.get_world_bank_data()
        status['world_bank'] = {
            'loaded': len(world_bank_data) > 0,
            'indicators': len(world_bank_data)
        }
        
        gold_prices = self.get_gold_prices()
        status['gold'] = {
            'loaded': gold_prices.get('ounce', 0) > 0,
            'price': gold_prices.get('ounce', 0)
        }
        
        return status