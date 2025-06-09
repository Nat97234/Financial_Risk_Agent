import streamlit as st
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
import os
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import yfinance as yf
from config.settings import *

logger = logging.getLogger(__name__)

class ExpertFinancialTools:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore
        self.session = requests.Session()
        try:
            self.search_tool = DuckDuckGoSearchRun()
        except Exception as e:
            logger.warning(f"Web search tool initialization failed: {e}")
            self.search_tool = None

    def financial_database_search(self, query: str) -> str:
        """Advanced search of internal financial database with relevance scoring"""
        if not self.vectorstore:
            return "Internal financial database not available. Please ensure data has been loaded."
        
        try:
            # Enhanced search with metadata filtering
            docs = self.vectorstore.similarity_search_with_score(query, k=10)
            
            if not docs:
                return f"No relevant financial data found for query: {query}"
            
            # Process and rank results
            results = []
            for doc, score in docs:
                # Add relevance score and metadata info
                source = doc.metadata.get('source', 'unknown')
                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                
                results.append({
                    'content': content_preview,
                    'source': source,
                    'relevance_score': f"{(1-score)*100:.1f}%",
                    'metadata': doc.metadata
                })
            
            # Format results for AI consumption
            formatted_results = []
            for i, result in enumerate(results[:5], 1):  # Top 5 results
                formatted_results.append(
                    f"Result {i} (Relevance: {result['relevance_score']}, Source: {result['source']}):\n"
                    f"{result['content']}\n"
                )
            
            return "Internal Financial Database Results:\n\n" + "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error in financial database search: {e}")
            return f"Error searching financial database: {str(e)}"

    def advanced_web_research(self, query: str) -> str:
        """Comprehensive web research for financial information"""
        if not self.search_tool:
            return "Web research tool not available"
        
        try:
            # Enhance query for financial context
            financial_query = f"financial analysis {query} 2024 2025 market data investment"
            
            # Get search results
            search_results = self.search_tool.run(financial_query)
            
            # Additional targeted searches for comprehensive coverage
            additional_queries = [
                f"{query} expert analysis 2024",
                f"{query} market outlook forecast",
                f"{query} investment recommendations"
            ]
            
            all_results = [f"Primary Search Results:\n{search_results}"]
            
            for additional_query in additional_queries[:2]:  # Limit to avoid rate limits
                try:
                    additional_result = self.search_tool.run(additional_query)
                    all_results.append(f"\nAdditional Research ({additional_query}):\n{additional_result}")
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Additional search failed: {e}")
                    continue
            
            return "\n".join(all_results)
            
        except Exception as e:
            logger.error(f"Error in web research: {e}")
            return f"Web research encountered an error: {str(e)}"

    def calculate_advanced_financial_metrics(self, data: str) -> str:
        """Calculate comprehensive financial metrics and ratios"""
        try:
            data_lower = data.lower()
            
            # Risk-adjusted returns
            if any(term in data_lower for term in ["sharpe", "risk adjusted", "volatility"]):
                return """
                **Risk-Adjusted Return Metrics:**
                
                1. **Sharpe Ratio**: (Return - Risk-free Rate) / Standard Deviation
                   - Measures excess return per unit of risk
                   - Ratios > 1.0 are generally good, > 2.0 are excellent
                
                2. **Sortino Ratio**: (Return - Risk-free Rate) / Downside Deviation
                   - Similar to Sharpe but only considers downside volatility
                   - Better measure for asymmetric return distributions
                
                3. **Treynor Ratio**: (Return - Risk-free Rate) / Beta
                   - Risk-adjusted return relative to systematic risk
                   - Useful for comparing portfolio performance
                
                4. **Information Ratio**: (Portfolio Return - Benchmark Return) / Tracking Error
                   - Measures active return per unit of active risk
                """
            
            # Valuation metrics
            elif any(term in data_lower for term in ["valuation", "p/e", "price", "fair value"]):
                return """
                **Comprehensive Valuation Metrics:**
                
                1. **Price-to-Earnings Ratios:**
                   - P/E Ratio: Market Price / Earnings Per Share
                   - Forward P/E: Uses projected earnings
                   - PEG Ratio: P/E / Earnings Growth Rate
                
                2. **Enterprise Value Metrics:**
                   - EV/EBITDA: Enterprise Value / Earnings Before Interest, Taxes, Depreciation, Amortization
                   - EV/Sales: Enterprise Value / Revenue
                
                3. **Book Value Metrics:**
                   - P/B Ratio: Market Price / Book Value per Share
                   - Price-to-Tangible Book Value
                
                4. **Discounted Cash Flow (DCF):**
                   - NPV = Σ(Cash Flow / (1 + Discount Rate)^n)
                   - Terminal Value = Final Year Cash Flow × (1 + Growth Rate) / (Discount Rate - Growth Rate)
                """
            
            # Portfolio optimization
            elif any(term in data_lower for term in ["portfolio", "optimization", "allocation", "diversification"]):
                return """
                **Portfolio Optimization & Risk Management:**
                
                1. **Modern Portfolio Theory:**
                   - Expected Return: Σ(Weight × Expected Return)
                   - Portfolio Variance: Σ(Wi² × σi²) + ΣΣ(Wi × Wj × σij)
                   - Efficient Frontier: Optimal risk-return combinations
                
                2. **Risk Metrics:**
                   - Value at Risk (VaR): Maximum expected loss at given confidence level
                   - Conditional VaR (CVaR): Expected loss beyond VaR threshold
                   - Maximum Drawdown: Peak-to-trough decline
                
                3. **Correlation Analysis:**
                   - Correlation Matrix: Measures relationships between assets
                   - Diversification Ratio: Risk reduction through diversification
                
                4. **Asset Allocation Models:**
                   - Strategic Asset Allocation: Long-term target weights
                   - Tactical Asset Allocation: Short-term adjustments
                   - Risk Parity: Equal risk contribution from all assets
                """
            
            # Options and derivatives
            elif any(term in data_lower for term in ["options", "derivatives", "black scholes", "volatility"]):
                return """
                **Options & Derivatives Valuation:**
                
                1. **Black-Scholes Model:**
                   - Call Option: C = S₀N(d₁) - Ke^(-rT)N(d₂)
                   - Put Option: P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
                   - Where d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
                
                2. **Greeks:**
                   - Delta: Price sensitivity to underlying asset
                   - Gamma: Rate of change of delta
                   - Theta: Time decay
                   - Vega: Volatility sensitivity
                   - Rho: Interest rate sensitivity
                
                3. **Implied Volatility:**
                   - Market's expectation of future volatility
                   - Volatility smile/skew patterns
                """
            
            # Credit analysis
            elif any(term in data_lower for term in ["credit", "bonds", "default", "rating"]):
                return """
                **Credit Analysis & Fixed Income:**
                
                1. **Credit Metrics:**
                   - Probability of Default (PD)
                   - Loss Given Default (LGD)
                   - Exposure at Default (EAD)
                   - Expected Loss = PD × LGD × EAD
                
                2. **Bond Pricing:**
                   - Present Value of Cash Flows
                   - Yield to Maturity (YTM)
                   - Duration and Convexity
                   - Credit Spread Analysis
                
                3. **Rating Analysis:**
                   - Credit rating migration probabilities
                   - Default correlation analysis
                """
            
            else:
                return """
                **Available Financial Calculations:**
                
                • Risk-Adjusted Returns (Sharpe, Sortino, Treynor ratios)
                • Valuation Metrics (P/E, EV/EBITDA, DCF models)
                • Portfolio Optimization (MPT, VaR, diversification)
                • Options Pricing (Black-Scholes, Greeks)
                • Credit Analysis (Default probability, bond pricing)
                • Technical Indicators (RSI, MACD, Bollinger Bands)
                • Economic Indicators Analysis
                
                Please specify which type of calculation you need, and I'll provide detailed formulas and analysis.
                """
                
        except Exception as e:
            logger.error(f"Error in financial calculations: {e}")
            return f"Financial calculation error: {str(e)}"

    def market_sentiment_analysis(self, query: str) -> str:
        """Advanced market sentiment analysis using multiple indicators"""
        try:
            # Get VIX data for fear/greed analysis
            vix_data = self._get_vix_data()
            
            # Get market breadth indicators
            market_breadth = self._get_market_breadth()
            
            # Web sentiment search
            sentiment_search = None
            if self.search_tool:
                sentiment_query = f"market sentiment {query} investor opinion fear greed index 2024"
                sentiment_search = self.search_tool.run(sentiment_query)
            
            sentiment_analysis = [
                "**Market Sentiment Analysis:**\n"
            ]
            
            # VIX interpretation
            if vix_data:
                sentiment_analysis.append(f"**Fear & Greed Indicator (VIX):**")
                sentiment_analysis.append(f"Current VIX Level: {vix_data['current']:.2f}")
                sentiment_analysis.append(f"Interpretation: {vix_data['interpretation']}")
                sentiment_analysis.append(f"Historical Context: {vix_data['context']}")
                sentiment_analysis.append("")
            
            # Market breadth
            if market_breadth:
                sentiment_analysis.append("**Market Breadth Indicators:**")
                for indicator, value in market_breadth.items():
                    sentiment_analysis.append(f"- {indicator}: {value}")
                sentiment_analysis.append("")
            
            # Web sentiment
            if sentiment_search:
                sentiment_analysis.append("**Current Market Sentiment from News & Analysis:**")
                sentiment_analysis.append(sentiment_search)
            
            # Technical sentiment indicators
            sentiment_analysis.append(self._get_technical_sentiment(query))
            
            return "\n".join(sentiment_analysis)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return f"Market sentiment analysis error: {str(e)}"

    def _get_vix_data(self) -> Optional[Dict[str, Any]]:
        """Get VIX data for fear/greed analysis"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1mo")
            
            if hist.empty:
                return None
            
            current_vix = hist['Close'].iloc[-1]
            avg_vix = hist['Close'].mean()
            
            # Interpretation
            if current_vix < 15:
                interpretation = "Low volatility - Market complacency, potential for surprise moves"
            elif current_vix < 25:
                interpretation = "Normal volatility - Stable market conditions"
            elif current_vix < 35:
                interpretation = "Elevated volatility - Increased uncertainty and risk"
            else:
                interpretation = "High volatility - Market stress, potential opportunities for contrarians"
            
            # Historical context
            if current_vix > avg_vix * 1.2:
                context = "Above recent average - Market more fearful than usual"
            elif current_vix < avg_vix * 0.8:
                context = "Below recent average - Market more complacent than usual"
            else:
                context = "Near recent average - Normal volatility levels"
            
            return {
                'current': current_vix,
                'average': avg_vix,
                'interpretation': interpretation,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return None

    def _get_market_breadth(self) -> Dict[str, str]:
        """Get market breadth indicators"""
        try:
            # Advance/Decline data (simplified - in production use more sophisticated data)
            breadth_indicators = {}
            
            # Get some market ETFs as proxies
            symbols = ['SPY', 'QQQ', 'IWM', 'VTI']  # Large, tech, small cap, total market
            
            total_positive = 0
            total_negative = 0
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if len(hist) >= 2:
                        change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                        if change > 0:
                            total_positive += 1
                        else:
                            total_negative += 1
                except:
                    continue
            
            if total_positive + total_negative > 0:
                breadth_indicators['Market Breadth'] = f"{total_positive} advancing, {total_negative} declining"
                
                if total_positive > total_negative:
                    breadth_indicators['Breadth Signal'] = "Positive - More sectors advancing"
                elif total_negative > total_positive:
                    breadth_indicators['Breadth Signal'] = "Negative - More sectors declining"
                else:
                    breadth_indicators['Breadth Signal'] = "Neutral - Mixed sector performance"
            
            return breadth_indicators
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            return {}

    def _get_technical_sentiment(self, query: str) -> str:
        """Get technical sentiment indicators"""
        try:
            # If query mentions specific stocks, analyze them
            symbols = self._extract_symbols_from_query(query)
            
            if not symbols:
                symbols = ['SPY']  # Default to market
            
            technical_sentiment = ["**Technical Sentiment Indicators:**"]
            
            for symbol in symbols[:3]:  # Limit to 3 symbols
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="3mo")
                    
                    if len(hist) < 50:
                        continue
                    
                    # RSI calculation
                    close = hist['Close']
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Moving averages
                    sma_20 = close.rolling(20).mean().iloc[-1]
                    sma_50 = close.rolling(50).mean().iloc[-1]
                    current_price = close.iloc[-1]
                    
                    # Technical signals
                    signals = []
                    if current_rsi > 70:
                        signals.append("Overbought (RSI > 70)")
                    elif current_rsi < 30:
                        signals.append("Oversold (RSI < 30)")
                    else:
                        signals.append(f"Neutral RSI ({current_rsi:.1f})")
                    
                    if current_price > sma_20 > sma_50:
                        signals.append("Bullish trend (Above moving averages)")
                    elif current_price < sma_20 < sma_50:
                        signals.append("Bearish trend (Below moving averages)")
                    else:
                        signals.append("Mixed trend signals")
                    
                    technical_sentiment.append(f"\n**{symbol}:**")
                    for signal in signals:
                        technical_sentiment.append(f"- {signal}")
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    continue
            
            return "\n".join(technical_sentiment)
            
        except Exception as e:
            logger.error(f"Error in technical sentiment: {e}")
            return "Technical sentiment analysis unavailable"

    def _extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract stock symbols from query"""
        import re
        
        # Common stock symbol patterns
        symbols = []
        
        # Look for explicit mentions of tickers (2-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, query.upper())
        
        # Filter out common words that aren't tickers
        common_words = {'THE', 'AND', 'FOR', 'YOU', 'ARE', 'WITH', 'THIS', 'THAT', 'FROM', 'HAVE', 'MORE', 'WILL', 'YOUR', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'NOT', 'BUT', 'ALL', 'WHY', 'USE', 'WAY', 'SHE', 'MAN', 'TOO', 'ANY', 'PUT', 'SAY', 'SAW', 'SET', 'TRY', 'ASK', 'BAD', 'BIG', 'END', 'FAR', 'FEW', 'GOT', 'LET', 'LOW', 'OFF', 'RUN', 'TOP', 'WIN', 'YES'}
        
        for ticker in potential_tickers:
            if ticker not in common_words and len(ticker) <= 5:
                symbols.append(ticker)
        
        return symbols[:5]  # Limit to 5 symbols

    def economic_data_analysis(self, query: str) -> str:
        """Analyze economic data and indicators"""
        try:
            analysis = ["**Economic Data Analysis:**\n"]
            
            # Get treasury yield data
            treasury_data = self._get_treasury_data()
            if treasury_data:
                analysis.append("**Interest Rate Environment:**")
                analysis.append(f"10-Year Treasury Yield: {treasury_data['10y']:.2f}%")
                analysis.append(f"2-Year Treasury Yield: {treasury_data['2y']:.2f}%")
                analysis.append(f"Yield Curve: {treasury_data['curve_status']}")
                analysis.append("")
            
            # Dollar strength
            dollar_data = self._get_dollar_index()
            if dollar_data:
                analysis.append("**Dollar Strength:**")
                analysis.append(f"DXY Index: {dollar_data['current']:.2f}")
                analysis.append(f"Trend: {dollar_data['trend']}")
                analysis.append("")
            
            # Inflation indicators
            analysis.append("**Inflation Considerations:**")
            analysis.append("- Monitor CPI and PCE data for Fed policy implications")
            analysis.append("- Consider TIPS (inflation-protected securities) if inflation rising")
            analysis.append("- Real assets (commodities, REITs) may benefit from inflation")
            analysis.append("")
            
            # Economic web search
            if self.search_tool:
                econ_query = f"economic indicators {query} GDP inflation employment 2024"
                econ_search = self.search_tool.run(econ_query)
                analysis.append("**Current Economic Analysis:**")
                analysis.append(econ_search)
            
            return "\n".join(analysis)
            
        except Exception as e:
            logger.error(f"Error in economic analysis: {e}")
            return f"Economic analysis error: {str(e)}"

    def _get_treasury_data(self) -> Optional[Dict[str, Any]]:
        """Get treasury yield data"""
        try:
            # Get 10-year and 2-year treasury yields
            ten_year = yf.Ticker("^TNX")
            two_year = yf.Ticker("^IRX")
            
            ten_year_hist = ten_year.history(period="5d")
            two_year_hist = two_year.history(period="5d")
            
            if ten_year_hist.empty or two_year_hist.empty:
                return None
            
            ten_year_yield = ten_year_hist['Close'].iloc[-1]
            two_year_yield = two_year_hist['Close'].iloc[-1]
            
            # Determine yield curve status
            curve_spread = ten_year_yield - two_year_yield
            
            if curve_spread > 1:
                curve_status = "Steep (Normal)"
            elif curve_spread > 0:
                curve_status = "Normal"
            elif curve_spread > -0.5:
                curve_status = "Flat"
            else:
                curve_status = "Inverted (Recession signal)"
            
            return {
                '10y': ten_year_yield,
                '2y': two_year_yield,
                'spread': curve_spread,
                'curve_status': curve_status
            }
            
        except Exception as e:
            logger.error(f"Error getting treasury data: {e}")
            return None

    def _get_dollar_index(self) -> Optional[Dict[str, Any]]:
        """Get dollar index data"""
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="1mo")
            
            if hist.empty:
                return None
            
            current = hist['Close'].iloc[-1]
            month_ago = hist['Close'].iloc[0]
            
            trend = "Strengthening" if current > month_ago else "Weakening"
            
            return {
                'current': current,
                'trend': trend,
                'change_pct': ((current - month_ago) / month_ago) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting dollar index: {e}")
            return None


class EnhancedFinancialAIAgent:
    def __init__(self):
        self.vectorstore = None
        self.agent_executor = None
        self.memory = ConversationBufferWindowMemory(
            k=15,  # Increased memory
            return_messages=True,
            memory_key="chat_history"
        )
        self.conversation_context = []
        
        if OPENAI_API_KEY:
            self.initialize_agent()

    def create_enhanced_vectorstore(self, documents: List[Document]) -> Optional[Any]:
        """Create enhanced vector store with better chunking and embeddings"""
        if not documents or not OPENAI_API_KEY:
            return None

        try:
            # Enhanced text splitting for financial documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Larger chunks for better context
                chunk_overlap=200,  # More overlap for continuity
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Enhanced embeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-large"  # Use latest embedding model
            )
            
            # Check for existing index
            if os.path.exists("faiss_index"):
                try:
                    vectorstore = FAISS.load_local(
                        "faiss_index",
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True
                    )
                    # Add new documents to existing index
                    vectorstore.add_documents(split_docs)
                    vectorstore.save_local("faiss_index")
                    logger.info(f"Updated existing vectorstore with {len(split_docs)} documents")
                except Exception as e:
                    logger.warning(f"Could not load existing index, creating new one: {e}")
                    vectorstore = FAISS.from_documents(split_docs, embeddings)
                    vectorstore.save_local("faiss_index")
            else:
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                vectorstore.save_local("faiss_index")
                logger.info(f"Created new vectorstore with {len(split_docs)} documents")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            return None

    def initialize_agent(self):
        """Initialize the enhanced AI agent with expert financial tools"""
        if not OPENAI_API_KEY:
            st.error("OpenAI API key is required for AI agent functionality")
            return

        try:
            # Use latest GPT-4 model
            llm = ChatOpenAI(
                model_name=AI_MODEL_CONFIG["PRIMARY_MODEL"],
                temperature=AI_MODEL_CONFIG["TEMPERATURE"],
                max_tokens=AI_MODEL_CONFIG["MAX_TOKENS"],
                openai_api_key=OPENAI_API_KEY
            )

            # Initialize expert financial tools
            financial_tools = ExpertFinancialTools(self.vectorstore)

            # Enhanced tool set
            tools = [
                Tool(
                    name="ExpertFinancialDatabase",
                    func=financial_tools.financial_database_search,
                    description="Search comprehensive financial database for stocks, bonds, economic data, company fundamentals, and market analysis. Use for historical data, financial statements, ratios, and market trends."
                ),
                Tool(
                    name="AdvancedWebResearch",
                    func=financial_tools.advanced_web_research,
                    description="Conduct comprehensive web research for current financial news, market analysis, expert opinions, and real-time market data. Use for breaking news, recent developments, and current market sentiment."
                ),
                Tool(
                    name="ExpertFinancialCalculations",
                    func=financial_tools.calculate_advanced_financial_metrics,
                    description="Perform advanced financial calculations including risk-adjusted returns, valuation models, portfolio optimization, options pricing, and credit analysis. Specify the type of calculation needed."
                ),
                Tool(
                    name="MarketSentimentAnalysis",
                    func=financial_tools.market_sentiment_analysis,
                    description="Analyze comprehensive market sentiment using VIX, technical indicators, market breadth, and current news sentiment. Use for understanding market psychology and positioning."
                ),
                Tool(
                    name="EconomicDataAnalysis",
                    func=financial_tools.economic_data_analysis,
                    description="Analyze economic indicators, interest rates, inflation data, and macroeconomic trends. Use for understanding economic environment and policy implications."
                )
            ]

            # Enhanced system prompt for financial expertise
            system_prompt = """You are an Expert Financial AI Agent with deep knowledge in:

**CORE EXPERTISE:**
- Investment Analysis & Portfolio Management
- Risk Assessment & Management
- Financial Planning & Strategy
- Market Analysis & Economics
- Corporate Finance & Valuation
- Behavioral Finance & Psychology

**YOUR CAPABILITIES:**
1. **Comprehensive Analysis**: Always use multiple tools to provide thorough analysis
2. **Risk-First Approach**: Always consider risk before returns
3. **Personalized Advice**: Tailor recommendations to user's profile and goals
4. **Evidence-Based**: Support all recommendations with data and research
5. **Multilingual**: Respond in user's preferred language (English/Arabic)

**ANALYSIS FRAMEWORK:**
For every financial question, follow this structure:
1. **Situation Analysis**: Understand the context and user needs
2. **Data Gathering**: Use financial database and web research tools
3. **Risk Assessment**: Evaluate potential risks and volatility
4. **Market Context**: Consider current market conditions and sentiment
5. **Quantitative Analysis**: Use appropriate financial calculations
6. **Personalized Recommendations**: Provide specific, actionable advice
7. **Implementation Strategy**: Explain how to execute recommendations
8. **Monitoring Plan**: Suggest how to track and adjust strategy

**QUALITY STANDARDS:**
- Always verify information using multiple sources
- Provide specific, actionable recommendations
- Include risk warnings and disclaimers
- Consider tax implications and costs
- Address both short-term and long-term perspectives
- Use professional financial terminology appropriately

**RESPONSE REQUIREMENTS:**
- Start with user profile acknowledgment if available
- Use clear headings and structured format
- Include specific numbers, percentages, and timeframes
- Provide alternative scenarios when appropriate
- End with clear next steps and implementation guidance

Remember: You are providing educational information, not personalized financial advice. Users should consult with qualified financial advisors for their specific situations."""

            # Create enhanced prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("assistant", "I'll provide comprehensive financial analysis using my expert tools and knowledge. Let me gather the relevant data and provide you with detailed insights.\n\n{agent_scratchpad}")
            ])

            # Create ReAct agent
            try:
                react_prompt = hub.pull("hwchase17/react")
                # Customize the react prompt for financial context
                react_prompt.messages[0].prompt.template = system_prompt + "\n\n" + react_prompt.messages[0].prompt.template
            except Exception:
                react_prompt = prompt_template

            agent = create_react_agent(llm, tools, react_prompt)
            
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=8,  # Increased for thorough analysis
                memory=self.memory,
                handle_parsing_errors=True,
                early_stopping_method="generate",
                return_intermediate_steps=True
            )

            logger.info("Enhanced Financial AI Agent initialized successfully")

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            st.error(f"Agent initialization failed: {e}")
            self.agent_executor = None
            
# Add this method to your EnhancedFinancialAIAgent class in ai_agent.py

def initialize_agent(self):
    """Initialize the enhanced AI agent with proper prompt template"""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is required for AI agent functionality")
        return

    try:
        # Use latest GPT-4 model
        llm = ChatOpenAI(
            model_name=AI_MODEL_CONFIG["PRIMARY_MODEL"],
            temperature=AI_MODEL_CONFIG["TEMPERATURE"],
            max_tokens=AI_MODEL_CONFIG["MAX_TOKENS"],
            openai_api_key=OPENAI_API_KEY
        )

        # Initialize expert financial tools
        financial_tools = ExpertFinancialTools(self.vectorstore)

        # Enhanced tool set
        tools = [
            Tool(
                name="ExpertFinancialDatabase",
                func=financial_tools.financial_database_search,
                description="Search comprehensive financial database for stocks, bonds, economic data, company fundamentals, and market analysis. Use for historical data, financial statements, ratios, and market trends."
            ),
            Tool(
                name="AdvancedWebResearch",
                func=financial_tools.advanced_web_research,
                description="Conduct comprehensive web research for current financial news, market analysis, expert opinions, and real-time market data. Use for breaking news, recent developments, and current market sentiment."
            ),
            Tool(
                name="ExpertFinancialCalculations",
                func=financial_tools.calculate_advanced_financial_metrics,
                description="Perform advanced financial calculations including risk-adjusted returns, valuation models, portfolio optimization, options pricing, and credit analysis. Specify the type of calculation needed."
            ),
            Tool(
                name="MarketSentimentAnalysis",
                func=financial_tools.market_sentiment_analysis,
                description="Analyze comprehensive market sentiment using VIX, technical indicators, market breadth, and current news sentiment. Use for understanding market psychology and positioning."
            ),
            Tool(
                name="EconomicDataAnalysis",
                func=financial_tools.economic_data_analysis,
                description="Analyze economic indicators, interest rates, inflation data, and macroeconomic trends. Use for understanding economic environment and policy implications."
            )
        ]

        # Create a proper prompt template
        from langchain_core.prompts import PromptTemplate

        template = """You are an Expert Financial AI Agent with deep knowledge in:

**CORE EXPERTISE:**
- Investment Analysis & Portfolio Management
- Risk Assessment & Management
- Financial Planning & Strategy
- Market Analysis & Economics
- Corporate Finance & Valuation
- Behavioral Finance & Psychology

**YOUR CAPABILITIES:**
1. **Comprehensive Analysis**: Always use multiple tools to provide thorough analysis
2. **Risk-First Approach**: Always consider risk before returns
3. **Personalized Advice**: Tailor recommendations to user's profile and goals
4. **Evidence-Based**: Support all recommendations with data and research
5. **Multilingual**: Respond in user's preferred language (English/Arabic)

**ANALYSIS FRAMEWORK:**
For every financial question, follow this structure:
1. **Situation Analysis**: Understand the context and user needs
2. **Data Gathering**: Use financial database and web research tools
3. **Risk Assessment**: Evaluate potential risks and volatility
4. **Market Context**: Consider current market conditions and sentiment
5. **Quantitative Analysis**: Use appropriate financial calculations
6. **Personalized Recommendations**: Provide specific, actionable advice
7. **Implementation Strategy**: Explain how to execute recommendations
8. **Monitoring Plan**: Suggest how to track and adjust strategy

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        # Create the prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": ", ".join([tool.name for tool in tools])
            }
        )

        # Create ReAct agent with proper prompt
        from langchain.agents import create_react_agent
        
        agent = create_react_agent(llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=8,
            memory=self.memory,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            return_intermediate_steps=True
        )

        logger.info("Enhanced Financial AI Agent initialized successfully")

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        st.error(f"Agent initialization failed: {e}")
        self.agent_executor = None