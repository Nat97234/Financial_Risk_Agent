import pandas as pd
import numpy as np
from langchain_core.documents import Document
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import streamlit as st
import logging
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import yfinance as yf
from config.settings import *

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedFinancialDataProcessor:
    def __init__(self):
        self.risk_free_rate = RISK_FREE_RATE
        self.market_return = MARKET_RETURN
        self.scaler = StandardScaler()
        
    def convert_data_to_documents(self, combined_data: Dict[str, Any]) -> List[Document]:
        """Convert various data sources to LangChain documents with enhanced processing"""
        documents = []
        
        try:
            # Process CSV data with enhanced financial context
            csv_data = combined_data.get("csv_data", pd.DataFrame())
            if not csv_data.empty:
                documents.extend(self._process_csv_to_documents(csv_data))
            
            # Process stock data with technical and fundamental analysis
            stocks_data = combined_data.get("stocks", {})
            if stocks_data:
                documents.extend(self._process_stocks_to_documents(stocks_data))
            
            # Process World Bank economic data
            world_bank_data = combined_data.get("world_bank", {})
            if world_bank_data:
                documents.extend(self._process_world_bank_to_documents(world_bank_data))
            
            # Process Alpha Vantage data
            alpha_data = combined_data.get("alpha_vantage", {})
            if alpha_data:
                documents.extend(self._process_alpha_vantage_to_documents(alpha_data))
            
            # Process FMP data
            fmp_data = combined_data.get("fmp", {})
            if fmp_data:
                documents.extend(self._process_fmp_to_documents(fmp_data))
            
            # Process cryptocurrency data
            crypto_data = combined_data.get("crypto", {})
            if crypto_data:
                documents.extend(self._process_crypto_to_documents(crypto_data))
            
            # Process forex data
            forex_data = combined_data.get("forex", {})
            if forex_data:
                documents.extend(self._process_forex_to_documents(forex_data))
            
            # Process commodities data
            commodities_data = combined_data.get("commodities", {})
            if commodities_data:
                documents.extend(self._process_commodities_to_documents(commodities_data))
            
            # Process economic indicators
            economic_indicators = combined_data.get("economic_indicators", {})
            if economic_indicators:
                documents.extend(self._process_economic_indicators_to_documents(economic_indicators))
            
            logger.info(f"Successfully processed {len(documents)} documents from all data sources")
            
        except Exception as e:
            logger.error(f"Error in convert_data_to_documents: {e}")
        
        return documents

    def _process_csv_to_documents(self, csv_data: pd.DataFrame) -> List[Document]:
        """Process CSV data with enhanced financial analysis"""
        documents = []
        
        if csv_data.empty:
            return documents
        
        try:
            # Limit rows for performance
            total_rows = len(csv_data)
            rows_to_process = min(total_rows, MAX_CSV_ROWS_FOR_PROCESSING)
            
            if rows_to_process < total_rows:
                # Strategic sampling for financial data
                csv_sample = self._strategic_sampling(csv_data, rows_to_process)
            else:
                csv_sample = csv_data
            
            # Enhanced data analysis
            data_analysis = self._analyze_financial_csv(csv_sample)
            
            # Create summary document
            summary_content = self._create_csv_summary_content(csv_sample, data_analysis)
            documents.append(Document(
                page_content=summary_content,
                metadata={
                    "source": "csv_summary",
                    "type": "financial_analysis",
                    "total_rows": total_rows,
                    "processed_rows": rows_to_process
                }
            ))
            
            # Process individual records with financial context
            chunks = [csv_sample[i:i + CSV_CHUNK_SIZE] for i in range(0, len(csv_sample), CSV_CHUNK_SIZE)]
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_documents = self._process_csv_chunk(chunk, chunk_idx)
                documents.extend(chunk_documents)
            
            st.success(f"✅ Processed {len(documents)} documents from CSV data ({rows_to_process:,} rows)")
            
        except Exception as e:
            logger.error(f"Error processing CSV to documents: {e}")
            st.error(f"Error processing CSV data: {e}")
        
        return documents

    def _strategic_sampling(self, df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
        """Strategic sampling for financial data to maintain representativeness"""
        try:
            # Try to identify time-based columns
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                # Time-based sampling to maintain temporal distribution
                df_sorted = df.sort_values(by=date_columns[0])
                step = len(df) // target_rows
                sampled_df = df_sorted.iloc[::step][:target_rows]
            else:
                # Random sampling with stratification if possible
                if len(df) > target_rows:
                    sampled_df = df.sample(n=target_rows, random_state=42)
                else:
                    sampled_df = df
            
            return sampled_df
            
        except Exception as e:
            logger.warning(f"Strategic sampling failed, using random sampling: {e}")
            return df.sample(n=min(target_rows, len(df)), random_state=42)

    def _analyze_financial_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive financial analysis of CSV data"""
        analysis = {}
        
        try:
            # Basic statistics
            analysis['basic_stats'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'text_columns': len(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().sum()
            }
            
            # Financial metrics identification
            financial_columns = self._identify_financial_columns(df)
            analysis['financial_columns'] = financial_columns
            
            # Statistical analysis for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                analysis['correlations'] = self._calculate_correlations(numeric_df)
                analysis['outliers'] = self._detect_outliers(numeric_df)
                analysis['distributions'] = self._analyze_distributions(numeric_df)
            
            # Time series analysis if applicable
            time_analysis = self._analyze_time_series(df)
            if time_analysis:
                analysis['time_series'] = time_analysis
            
        except Exception as e:
            logger.error(f"Error in financial CSV analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis

    def _identify_financial_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify different types of financial columns"""
        financial_columns = {
            'price_columns': [],
            'volume_columns': [],
            'ratio_columns': [],
            'return_columns': [],
            'risk_columns': [],
            'fundamental_columns': []
        }
        
        try:
            for col in df.columns:
                col_lower = col.lower()
                
                # Price-related columns
                if any(term in col_lower for term in ['price', 'close', 'open', 'high', 'low', 'value']):
                    financial_columns['price_columns'].append(col)
                
                # Volume-related columns
                elif any(term in col_lower for term in ['volume', 'shares', 'quantity']):
                    financial_columns['volume_columns'].append(col)
                
                # Ratio columns
                elif any(term in col_lower for term in ['ratio', 'pe', 'pb', 'roe', 'roa', 'margin']):
                    financial_columns['ratio_columns'].append(col)
                
                # Return columns
                elif any(term in col_lower for term in ['return', 'yield', 'growth', 'change']):
                    financial_columns['return_columns'].append(col)
                
                # Risk columns
                elif any(term in col_lower for term in ['volatility', 'beta', 'risk', 'deviation']):
                    financial_columns['risk_columns'].append(col)
                
                # Fundamental columns
                elif any(term in col_lower for term in ['revenue', 'earnings', 'profit', 'debt', 'assets']):
                    financial_columns['fundamental_columns'].append(col)
        
        except Exception as e:
            logger.error(f"Error identifying financial columns: {e}")
        
        return financial_columns

    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation analysis for financial data"""
        try:
            correlation_matrix = df.corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > RISK_METRICS['CORRELATION_THRESHOLD']:
                        high_correlations.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            return {
                'matrix_shape': correlation_matrix.shape,
                'high_correlations': high_correlations,
                'max_correlation': correlation_matrix.values.max(),
                'min_correlation': correlation_matrix.values.min()
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        outliers = {}
        
        try:
            for column in df.columns:
                if df[column].dtype in [np.number]:
                    # IQR method
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
                    
                    if outlier_count > 0:
                        outliers[column] = {
                            'count': outlier_count,
                            'percentage': (outlier_count / len(df)) * 100,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }
        
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
        
        return outliers

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical distributions of financial data"""
        distributions = {}
        
        try:
            for column in df.columns:
                if df[column].dtype in [np.number] and not df[column].isnull().all():
                    # Basic distribution statistics
                    skewness = stats.skew(df[column].dropna())
                    kurtosis = stats.kurtosis(df[column].dropna())
                    
                    # Normality test (Shapiro-Wilk for small samples)
                    if len(df[column].dropna()) < 5000:
                        _, p_value = stats.shapiro(df[column].dropna())
                        is_normal = p_value > 0.05
                    else:
                        # Use Kolmogorov-Smirnov for large samples
                        _, p_value = stats.kstest(df[column].dropna(), 'norm')
                        is_normal = p_value > 0.05
                    
                    distributions[column] = {
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'is_normal': is_normal,
                        'p_value': p_value
                    }
        
        except Exception as e:
            logger.error(f"Error analyzing distributions: {e}")
        
        return distributions

    def _analyze_time_series(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze time series patterns in financial data"""
        try:
            # Look for date columns
            date_columns = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    date_columns.append(col)
            
            if not date_columns:
                return None
            
            # Use the first date column found
            date_col = date_columns[0]
            
            # Basic time series analysis
            time_analysis = {
                'date_column': date_col,
                'date_range': {
                    'start': df[date_col].min() if pd.notna(df[date_col].min()) else None,
                    'end': df[date_col].max() if pd.notna(df[date_col].max()) else None
                },
                'frequency': self._infer_frequency(df[date_col])
            }
            
            return time_analysis
            
        except Exception as e:
            logger.warning(f"Time series analysis failed: {e}")
            return None

    def _infer_frequency(self, date_series: pd.Series) -> str:
        """Infer the frequency of time series data"""
        try:
            if len(date_series) < 2:
                return "Unknown"
            
            # Calculate differences between consecutive dates
            date_series_sorted = date_series.sort_values().dropna()
            differences = date_series_sorted.diff().dropna()
            
            # Get the most common difference
            mode_diff = differences.mode()
            if len(mode_diff) > 0:
                days_diff = mode_diff.iloc[0].days
                
                if days_diff == 1:
                    return "Daily"
                elif days_diff == 7:
                    return "Weekly"
                elif 28 <= days_diff <= 31:
                    return "Monthly"
                elif 88 <= days_diff <= 95:
                    return "Quarterly"
                elif 360 <= days_diff <= 370:
                    return "Annual"
                else:
                    return f"Every {days_diff} days"
            
            return "Irregular"
            
        except Exception as e:
            logger.warning(f"Frequency inference failed: {e}")
            return "Unknown"

    def _create_csv_summary_content(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Create comprehensive summary content for CSV data"""
        try:
            content_parts = [
                "Financial Dataset Analysis Summary",
                "=" * 40,
                f"Dataset Overview:",
                f"- Total Records: {len(df):,}",
                f"- Total Columns: {len(df.columns)}",
                f"- Numeric Columns: {analysis.get('basic_stats', {}).get('numeric_columns', 0)}",
                f"- Text Columns: {analysis.get('basic_stats', {}).get('text_columns', 0)}",
                f"- Missing Values: {analysis.get('basic_stats', {}).get('missing_values', 0):,}",
                ""
            ]
            
            # Financial columns summary
            financial_cols = analysis.get('financial_columns', {})
            if any(financial_cols.values()):
                content_parts.extend([
                    "Financial Data Structure:",
                    f"- Price Columns: {', '.join(financial_cols.get('price_columns', []))}" if financial_cols.get('price_columns') else "",
                    f"- Volume Columns: {', '.join(financial_cols.get('volume_columns', []))}" if financial_cols.get('volume_columns') else "",
                    f"- Financial Ratios: {', '.join(financial_cols.get('ratio_columns', []))}" if financial_cols.get('ratio_columns') else "",
                    f"- Return Metrics: {', '.join(financial_cols.get('return_columns', []))}" if financial_cols.get('return_columns') else "",
                    ""
                ])
            
            # Statistical insights
            correlations = analysis.get('correlations', {})
            if correlations.get('high_correlations'):
                content_parts.extend([
                    "Key Correlations (>70%):",
                ])
                for corr in correlations['high_correlations'][:5]:  # Top 5
                    content_parts.append(f"- {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f}")
                content_parts.append("")
            
            # Outlier summary
            outliers = analysis.get('outliers', {})
            if outliers:
                content_parts.extend([
                    "Outlier Detection:",
                ])
                for col, outlier_info in list(outliers.items())[:3]:  # Top 3
                    content_parts.append(f"- {col}: {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%)")
                content_parts.append("")
            
            # Time series info
            time_series = analysis.get('time_series')
            if time_series:
                content_parts.extend([
                    "Time Series Information:",
                    f"- Date Column: {time_series['date_column']}",
                    f"- Date Range: {time_series['date_range']['start']} to {time_series['date_range']['end']}",
                    f"- Frequency: {time_series['frequency']}",
                    ""
                ])
            
            # Data quality assessment
            content_parts.extend([
                "Data Quality Assessment:",
                f"- Completeness: {((len(df) * len(df.columns) - analysis.get('basic_stats', {}).get('missing_values', 0)) / (len(df) * len(df.columns)) * 100):.1f}%",
                f"- Numeric Data Ratio: {(analysis.get('basic_stats', {}).get('numeric_columns', 0) / len(df.columns) * 100):.1f}%",
                ""
            ])
            
            return "\n".join(filter(None, content_parts))
            
        except Exception as e:
            logger.error(f"Error creating CSV summary: {e}")
            return f"CSV Data Summary - {len(df)} records, {len(df.columns)} columns"

    def _process_csv_chunk(self, chunk: pd.DataFrame, chunk_idx: int) -> List[Document]:
        """Process individual CSV chunk with financial context"""
        documents = []
        
        try:
            # Group related records for better context
            for idx, row in chunk.iterrows():
                content_parts = []
                
                # Create meaningful content from row data
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        content_parts.append(f"{col}: {value}")
                
                if content_parts:
                    content = "\n".join(content_parts)
                    
                    # Add financial context if detectable
                    financial_context = self._add_financial_context(row)
                    if financial_context:
                        content += f"\n\nFinancial Context:\n{financial_context}"
                    
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": "csv_data",
                            "chunk_index": chunk_idx,
                            "row_index": idx,
                            "data_type": "financial_record"
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Error processing CSV chunk {chunk_idx}: {e}")
        
        return documents

    def _add_financial_context(self, row: pd.Series) -> str:
        """Add financial context to individual records"""
        context_parts = []
        
        try:
            # Look for financial indicators in the row
            for col, value in row.items():
                if pd.notna(value):
                    col_lower = col.lower()
                    
                    # Price analysis
                    if 'price' in col_lower and isinstance(value, (int, float)):
                        if value > 1000:
                            context_parts.append(f"High-value asset: ${value:,.2f}")
                        elif value < 1:
                            context_parts.append(f"Fractional pricing: ${value:.4f}")
                    
                    # Ratio analysis
                    elif any(term in col_lower for term in ['pe', 'pb', 'roe']):
                        if isinstance(value, (int, float)):
                            if 'pe' in col_lower and value > 30:
                                context_parts.append("High P/E ratio - potential growth stock")
                            elif 'pe' in col_lower and value < 10:
                                context_parts.append("Low P/E ratio - potential value stock")
            
            return "; ".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.warning(f"Error adding financial context: {e}")
            return ""

    def _process_stocks_to_documents(self, stocks_data: Dict[str, Any]) -> List[Document]:
        """Process stock data with comprehensive analysis"""
        documents = []
        
        try:
            for symbol, data in stocks_data.items():
                if not data:
                    continue
                
                # Company overview document
                company_doc = self._create_company_document(symbol, data)
                if company_doc:
                    documents.append(company_doc)
                
                # Technical analysis document
                technical_doc = self._create_technical_document(symbol, data)
                if technical_doc:
                    documents.append(technical_doc)
                
                # Fundamental analysis document
                fundamental_doc = self._create_fundamental_document(symbol, data)
                if fundamental_doc:
                    documents.append(fundamental_doc)
                
                # Performance analysis document
                performance_doc = self._create_performance_document(symbol, data)
                if performance_doc:
                    documents.append(performance_doc)
        
        except Exception as e:
            logger.error(f"Error processing stocks to documents: {e}")
        
        return documents

    def _create_company_document(self, symbol: str, data: Dict[str, Any]) -> Optional[Document]:
        """Create comprehensive company profile document"""
        try:
            info = data.get('info', {})
            if not info:
                return None
            
            content_parts = [
                f"Company Profile: {symbol}",
                "=" * 30,
                f"Company Name: {info.get('longName', 'N/A')}",
                f"Sector: {info.get('sector', 'N/A')}",
                f"Industry: {info.get('industry', 'N/A')}",
                f"Country: {info.get('country', 'N/A')}",
                f"Website: {info.get('website', 'N/A')}",
                "",
                "Business Overview:",
                f"{info.get('longBusinessSummary', 'No business summary available')}",
                "",
                "Key Metrics:",
                f"Market Cap: {self._format_currency(info.get('marketCap', 0))}",
                f"Enterprise Value: {self._format_currency(info.get('enterpriseValue', 0))}",
                f"Employees: {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "Employees: N/A",
                f"Exchange: {info.get('exchange', 'N/A')}",
                f"Currency: {info.get('currency', 'N/A')}",
                "",
                "Financial Highlights:",
                f"Revenue: {self._format_currency(info.get('totalRevenue', 0))}",
                f"Gross Profit: {self._format_currency(info.get('grossProfits', 0))}",
                f"EBITDA: {self._format_currency(info.get('ebitda', 0))}",
                f"Free Cash Flow: {self._format_currency(info.get('freeCashflow', 0))}",
                ""
            ]
            
            return Document(
                page_content="\n".join(content_parts),
                metadata={
                    "source": "company_profile",
                    "symbol": symbol,
                    "sector": info.get('sector', 'Unknown'),
                    "market_cap": info.get('marketCap', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating company document for {symbol}: {e}")
            return None

    def _format_currency(self, amount: float) -> str:
        """Format currency values for better readability"""
        if pd.isna(amount) or amount == 0:
            return "N/A"
        
        if abs(amount) >= 1e12:
            return f"${amount/1e12:.2f}T"
        elif abs(amount) >= 1e9:
            return f"${amount/1e9:.2f}B"
        elif abs(amount) >= 1e6:
            return f"${amount/1e6:.2f}M"
        elif abs(amount) >= 1e3:
            return f"${amount/1e3:.2f}K"
        else:
            return f"${amount:.2f}"

    # Additional placeholder methods for other document processors
    def _create_technical_document(self, symbol: str, data: Dict[str, Any]) -> Optional[Document]:
        """Create technical analysis document - simplified version"""
        try:
            technical = data.get('technical', {})
            history = data.get('history', pd.DataFrame())
            
            if technical or not history.empty:
                content_parts = [
                    f"Technical Analysis: {symbol}",
                    "=" * 30,
                ]
                
                # Current price info
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                    content_parts.extend([
                        f"Current Price: ${current_price:.2f}",
                        ""
                    ])
                
                return Document(
                    page_content="\n".join(content_parts),
                    metadata={
                        "source": "technical_analysis",
                        "symbol": symbol,
                        "analysis_type": "technical"
                    }
                )
        except Exception as e:
            logger.error(f"Error creating technical document for {symbol}: {e}")
            return None

    def _create_fundamental_document(self, symbol: str, data: Dict[str, Any]) -> Optional[Document]:
        """Create fundamental analysis document - simplified version"""
        try:
            info = data.get('info', {})
            if not info:
                return None
            
            content_parts = [
                f"Fundamental Analysis: {symbol}",
                "=" * 30,
                "Valuation Metrics:",
                f"P/E Ratio: {info.get('trailingPE', 'N/A')}",
                f"P/B Ratio: {info.get('priceToBook', 'N/A')}",
                ""
            ]
            
            return Document(
                page_content="\n".join(content_parts),
                metadata={
                    "source": "fundamental_analysis",
                    "symbol": symbol,
                    "analysis_type": "fundamental"
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating fundamental document for {symbol}: {e}")
            return None

    def _create_performance_document(self, symbol: str, data: Dict[str, Any]) -> Optional[Document]:
        """Create performance analysis document - simplified version"""
        try:
            history = data.get('history', pd.DataFrame())
            if history.empty:
                return None
            
            content_parts = [
                f"Performance Analysis: {symbol}",
                "=" * 30,
                f"Price Range: ${history['Low'].min():.2f} - ${history['High'].max():.2f}",
                ""
            ]
            
            return Document(
                page_content="\n".join(content_parts),
                metadata={
                    "source": "performance_analysis",
                    "symbol": symbol,
                    "analysis_type": "performance"
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating performance document for {symbol}: {e}")
            return None

    # Placeholder methods for other data processors (simplified for space)
    def _process_world_bank_to_documents(self, wb_data: Dict[str, Any]) -> List[Document]:
        """Process World Bank data to documents"""
        documents = []
        
        try:
            for indicator, data in wb_data.items():
                if data is not None and hasattr(data, 'to_string'):
                    content = f"World Bank Economic Data - {indicator.upper()}:\n\n{data.to_string()}"
                    
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": "world_bank",
                            "indicator": indicator,
                            "data_type": "economic"
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Error processing World Bank data: {e}")
        
        return documents

    def _process_alpha_vantage_to_documents(self, av_data: Dict[str, Any]) -> List[Document]:
        """Process Alpha Vantage data to documents"""
        documents = []
        
        try:
            for symbol, data in av_data.items():
                if not data:
                    continue
                
                content_parts = [f"Alpha Vantage Data - {symbol}"]
                
                if len(content_parts) > 1:
                    documents.append(Document(
                        page_content="\n".join(content_parts),
                        metadata={
                            "source": "alpha_vantage",
                            "symbol": symbol,
                            "data_type": "stock_data"
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data: {e}")
        
        return documents

    def _process_fmp_to_documents(self, fmp_data: Dict[str, Any]) -> List[Document]:
        """Process FMP data to documents"""
        documents = []
        
        try:
            for symbol, data in fmp_data.items():
                if not data:
                    continue
                
                content_parts = [f"Financial Modeling Prep Data - {symbol}"]
                
                documents.append(Document(
                    page_content="\n".join(content_parts),
                    metadata={
                        "source": "fmp",
                        "symbol": symbol,
                        "data_type": "comprehensive_analysis"
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error processing FMP data: {e}")
        
        return documents

    def _process_crypto_to_documents(self, crypto_data: Dict[str, Any]) -> List[Document]:
        """Process cryptocurrency data to documents"""
        documents = []
        
        try:
            for symbol, data in crypto_data.items():
                if not data:
                    continue
                
                content_parts = [
                    f"Cryptocurrency Analysis - {symbol}",
                    "=" * 30,
                    f"Current Price: ${data.get('current_price', 0):.4f}",
                    f"24h Volume: {data.get('volume_24h', 0):,.0f}",
                    f"Annualized Volatility: {data.get('volatility', 0)*100:.2f}%"
                ]
                
                documents.append(Document(
                    page_content="\n".join(content_parts),
                    metadata={
                        "source": "cryptocurrency",
                        "symbol": symbol,
                        "asset_class": "crypto"
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error processing crypto data: {e}")
        
        return documents

    def _process_forex_to_documents(self, forex_data: Dict[str, Any]) -> List[Document]:
        """Process forex data to documents"""
        documents = []
        
        try:
            for pair, data in forex_data.items():
                if not data:
                    continue
                
                content_parts = [
                    f"Forex Analysis - {pair}",
                    "=" * 30,
                    f"Current Rate: {data.get('current_rate', 0):.5f}",
                    f"Daily Change: {data.get('change_1d', 0):+.4f}%",
                    f"Volatility: {data.get('volatility', 0)*100:.2f}%"
                ]
                
                documents.append(Document(
                    page_content="\n".join(content_parts),
                    metadata={
                        "source": "forex",
                        "pair": pair,
                        "asset_class": "currency"
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error processing forex data: {e}")
        
        return documents

    def _process_commodities_to_documents(self, commodities_data: Dict[str, Any]) -> List[Document]:
        """Process commodities data to documents"""
        documents = []
        
        try:
            for commodity, data in commodities_data.items():
                if not data:
                    continue
                
                content_parts = [
                    f"Commodity Analysis - {commodity}",
                    "=" * 30,
                    f"Current Price: ${data.get('current_price', 0):.2f}",
                    f"YTD Change: {data.get('change_ytd', 0):+.2f}%",
                    f"Volatility: {data.get('volatility', 0)*100:.2f}%"
                ]
                
                documents.append(Document(
                    page_content="\n".join(content_parts),
                    metadata={
                        "source": "commodities",
                        "commodity": commodity,
                        "asset_class": "commodity"
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error processing commodities data: {e}")
        
        return documents

    def _process_economic_indicators_to_documents(self, indicators_data: Dict[str, Any]) -> List[Document]:
        """Process economic indicators to documents"""
        documents = []
        
        try:
            for indicator, data in indicators_data.items():
                if not data:
                    continue
                
                content_parts = [
                    f"Economic Indicator - {indicator.upper()}",
                    "=" * 30,
                    f"Current Value: {data.get('current', 'N/A')}",
                ]
                
                if 'interpretation' in data:
                    content_parts.extend([
                        "",
                        "Market Interpretation:",
                        data['interpretation']
                    ])
                
                documents.append(Document(
                    page_content="\n".join(content_parts),
                    metadata={
                        "source": "economic_indicators",
                        "indicator": indicator,
                        "data_type": "economic"
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error processing economic indicators: {e}")
        
        return documents


# Factory function for backward compatibility
def convert_data_to_documents(combined_data: Dict[str, Any]) -> List[Document]:
    """Factory function to maintain backward compatibility"""
    processor = EnhancedFinancialDataProcessor()
    return processor.convert_data_to_documents(combined_data)
                        