import pandas as pd
from langchain_core.documents import Document
from typing import Dict, List, Any
from config.settings import MAX_CSV_ROWS_FOR_PROCESSING, CSV_CHUNK_SIZE
import streamlit as st

def convert_data_to_documents(combined_data: Dict[str, Any]) -> List[Document]:
    """Convert various data sources to LangChain documents"""
    documents = []
    
    # Process CSV data with memory optimization
    csv_data = combined_data.get("csv_data", pd.DataFrame())
    if not csv_data.empty:
        documents.extend(process_csv_to_documents(csv_data))
    
    # Process yfinance data
    yfinance_data = combined_data.get("yfinance_data", {})
    if yfinance_data:
        for symbol, data in yfinance_data.items():
            if 'info' in data and data['info']:
                # Filter out None values and create content
                info_items = [f"{k}: {v}" for k, v in data['info'].items() if v is not None and str(v).strip()]
                if info_items:
                    content = f"Company: {symbol}\n" + "\n".join(info_items)
                    documents.append(Document(page_content=content, metadata={"source": "yfinance", "symbol": symbol}))
    
    # Process World Bank data
    world_bank_data = combined_data.get("world_bank_data", {})
    if world_bank_data:
        for indicator, data in world_bank_data.items():
            if hasattr(data, 'to_string'):
                content = f"World Bank {indicator} data:\n{data.to_string()}"
                documents.append(Document(page_content=content, metadata={"source": "world_bank", "indicator": indicator}))
    
    # Process Alpha Vantage data
    alpha_data = combined_data.get("alpha_data", {})
    if alpha_data:
        for symbol, data in alpha_data.items():
            if hasattr(data, 'to_string'):
                content = f"Alpha Vantage data for {symbol}:\n{data.tail(10).to_string()}"
                documents.append(Document(page_content=content, metadata={"source": "alpha_vantage", "symbol": symbol}))
    
    return documents

def process_csv_to_documents(csv_data: pd.DataFrame) -> List[Document]:
    """Process CSV data to documents with memory optimization for large datasets"""
    documents = []
    
    if csv_data.empty:
        return documents
    
    try:
        # Limit the number of rows processed to avoid memory issues
        total_rows = len(csv_data)
        rows_to_process = min(total_rows, MAX_CSV_ROWS_FOR_PROCESSING)
        
        if rows_to_process < total_rows:
            st.info(f"Processing {rows_to_process:,} out of {total_rows:,} CSV rows to optimize performance")
            # Sample the data to get representative rows
            csv_data_sample = csv_data.sample(n=rows_to_process, random_state=42)
        else:
            csv_data_sample = csv_data
        
        # Process in chunks to manage memory
        chunks = [csv_data_sample[i:i + CSV_CHUNK_SIZE] for i in range(0, len(csv_data_sample), CSV_CHUNK_SIZE)]
        
        for chunk_idx, chunk in enumerate(chunks):
            for idx, row in chunk.iterrows():
                # Create content from non-null values
                content_parts = []
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        content_parts.append(f"{col}: {value}")
                
                if content_parts:
                    content = "\n".join(content_parts)
                    documents.append(Document(
                        page_content=content, 
                        metadata={
                            "source": "csv", 
                            "row_index": idx,
                            "chunk": chunk_idx
                        }
                    ))
        
        st.success(f"✅ Processed {len(documents)} documents from CSV data")
        
    except Exception as e:
        st.error(f"Error processing CSV to documents: {e}")
    
    return documents

def process_combined_data(combined_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process and clean combined data from all sources"""
    processed_data = {}
    
    # Process stock data for analysis
    yfinance_data = combined_data.get("yfinance_data", {})
    if yfinance_data:
        stock_summary = []
        for symbol, data in yfinance_data.items():
            if 'info' in data and data['info']:
                info = data['info']
                stock_summary.append({
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'price': info.get('currentPrice', 0),
                    'market_cap': info.get('marketCap', 0),
                    'sector': info.get('sector', 'Unknown')
                })
        if stock_summary:
            processed_data['stock_summary'] = pd.DataFrame(stock_summary)
    
    # Process performance data
    if yfinance_data:
        performers = []
        for symbol, data in yfinance_data.items():
            if 'history' in data and len(data['history']) > 1:
                hist = data['history']
                if len(hist) >= 2:
                    try:
                        change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                        performers.append((symbol, change))
                    except (IndexError, ZeroDivisionError):
                        continue
        
        if performers:
            performers.sort(key=lambda x: x[1], reverse=True)
            processed_data['performers'] = performers
    
    # Process gold prices
    gold_prices = combined_data.get("gold_prices", {})
    if gold_prices and gold_prices.get('ounce', 0) > 0:
        processed_data['gold_prices'] = gold_prices
    
    # Process CSV data summary
    csv_data = combined_data.get("csv_data", pd.DataFrame())
    if not csv_data.empty:
        processed_data['csv_summary'] = {
            'total_rows': len(csv_data),
            'total_columns': len(csv_data.columns),
            'column_names': csv_data.columns.tolist(),
            'memory_usage_mb': csv_data.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    return processed_data

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare financial data"""
    if df.empty:
        return df
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Fill NaN values with appropriate defaults
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].fillna('Unknown')
    
    return df

def format_currency(amount: float) -> str:
    """Format currency values"""
    if amount >= 1e12:
        return f"${amount/1e12:.2f}T"
    elif amount >= 1e9:
        return f"${amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"${amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.2f}K"
    else:
        return f"${amount:.2f}"

def calculate_returns(prices: pd.Series) -> Dict[str, float]:
    """Calculate various return metrics"""
    if len(prices) < 2:
        return {}
    
    returns = prices.pct_change().dropna()
    
    return {
        'daily_return': returns.iloc[-1] * 100,
        'volatility': returns.std() * 100,
        'total_return': ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100,
        'sharpe_ratio': returns.mean() / returns.std() if returns.std() != 0 else 0
    }

def get_csv_statistics(csv_data: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive statistics about the CSV data"""
    if csv_data.empty:
        return {}
    
    stats = {
        'basic_info': {
            'total_rows': len(csv_data),
            'total_columns': len(csv_data.columns),
            'memory_usage_mb': csv_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': csv_data.duplicated().sum()
        },
        'data_types': csv_data.dtypes.value_counts().to_dict(),
        'missing_data': {
            'total_missing': csv_data.isnull().sum().sum(),
            'columns_with_missing': csv_data.columns[csv_data.isnull().any()].tolist(),
            'missing_percentage': (csv_data.isnull().sum() / len(csv_data) * 100).round(2).to_dict()
        }
    }
    
    # Add numeric column statistics
    numeric_columns = csv_data.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        stats['numeric_summary'] = csv_data[numeric_columns].describe().to_dict()
    
    return stats