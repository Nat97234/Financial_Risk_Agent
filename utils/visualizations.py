import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from config.settings import COMPANY_SYMBOLS

def create_dashboard(combined_data):
    """Create the main dashboard with multiple tabs"""
    st.markdown('<h2 class="section-header">📈 Interactive Financial Dashboard</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "🏢 Company Comparison", "🌍 Economic Indicators", "🔄 Real-time Data"])

    with tab1:
        create_stock_analysis_tab(combined_data)
    
    with tab2:
        create_company_comparison_tab(combined_data)
    
    with tab3:
        create_economic_indicators_tab(combined_data)
    
    with tab4:
        create_realtime_data_tab(combined_data)

def create_stock_analysis_tab(combined_data):
    """Create stock analysis visualization"""
    st.subheader("📊 Advanced Stock Analysis")
    
    selected_stocks = st.multiselect("Select stocks to analyze:", COMPANY_SYMBOLS, default=["AAPL", "MSFT", "TSLA"])
    
    yfinance_data = combined_data.get('yfinance_data', {})
    
    if selected_stocks and yfinance_data:
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=('Stock Prices', 'Trading Volume'),
                          vertical_spacing=0.1)
        
        colors = px.colors.qualitative.Set1
        
        for i, symbol in enumerate(selected_stocks):
            color = colors[i % len(colors)]
            
            if symbol in yfinance_data and 'history' in yfinance_data[symbol]:
                df = yfinance_data[symbol]['history']
                if not df.empty:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['Close'], name=f"{symbol} Price", 
                                 line=dict(color=color, width=2)),
                        row=1, col=1
                    )
                    if 'Volume' in df.columns:
                        fig.add_trace(
                            go.Bar(x=df.index, y=df['Volume'], name=f"{symbol} Volume", 
                                  marker_color=color, opacity=0.6),
                            row=2, col=1
                        )
        
        fig.update_layout(height=600, title_text="Advanced Stock Analysis Dashboard")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select stocks to view analysis or check if data is available.")

def create_company_comparison_tab(combined_data):
    """Create company comparison visualization"""
    st.subheader("🏢 Market Cap & Company Metrics")
    
    yfinance_data = combined_data.get('yfinance_data', {})
    
    if yfinance_data:
        market_caps = []
        companies = []
        sectors = []
        
        for symbol, data in yfinance_data.items():
            if 'info' in data and data['info'] and data['info'].get('marketCap'):
                market_caps.append(data['info']['marketCap'] / 1e9)
                companies.append(symbol)
                sectors.append(data['info'].get('sector', 'Unknown'))
        
        if market_caps:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=companies,
                y=market_caps,
                text=[f"${cap:.1f}B" for cap in market_caps],
                textposition='auto',
                marker_color=px.colors.qualitative.Set3,
                hovertemplate='<b>%{x}</b><br>Market Cap: $%{y:.1f}B<extra></extra>'
            ))
            
            fig.update_layout(
                title="Market Capitalization Comparison",
                xaxis_title="Company",
                yaxis_title="Market Cap (Billions USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Market cap data not available for comparison.")
    else:
        st.info("Company data not available.")

def create_economic_indicators_tab(combined_data):
    """Create economic indicators visualization"""
    st.subheader("🌍 Global Economic Indicators")
    
    world_bank_data = combined_data.get('world_bank_data', {})
    
    if world_bank_data:
        indicator = st.selectbox("Select Economic Indicator:", 
                                ["GDP", "Inflation", "Unemployment"], 
                                key="wb_indicator")
        
        if indicator.lower() in world_bank_data:
            data = world_bank_data[indicator.lower()]
            
            if hasattr(data, 'reset_index'):
                try:
                    df = data.reset_index()
                    
                    fig = go.Figure()
                    
                    for country in df.columns[1:]:
                        if country in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.iloc[:, 0],
                                y=df[country],
                                mode='lines+markers',
                                name=country,
                                line=dict(width=2)
                            ))
                    
                    fig.update_layout(
                        title=f"{indicator} Trends by Country",
                        xaxis_title="Year",
                        yaxis_title=f"{indicator} Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying {indicator} data: {e}")
        else:
            st.info(f"{indicator} data not available.")
    else:
        st.info("World Bank economic data not available.")

def create_realtime_data_tab(combined_data):
    """Create real-time data visualization"""
    st.subheader("🔄 Real-time Market Data")
    
    yfinance_data = combined_data.get('yfinance_data', {})
    gold_prices = combined_data.get('gold_prices', {})
    
    if yfinance_data:
        st.markdown("#### Market Summary")
        
        summary_data = []
        for symbol, data in list(yfinance_data.items())[:6]:
            if 'info' in data and data['info']:
                info = data['info']
                summary_data.append({
                    'Symbol': symbol,
                    'Company': info.get('longName', symbol)[:30] if info.get('longName') else symbol,
                    'Price': f"${info.get('currentPrice', 0):.2f}",
                    'Market Cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A",
                    'Sector': info.get('sector', 'Unknown')
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
        else:
            st.info("Market summary data not available.")
    
    # Market indices (sample data - in production, fetch real data)
    st.markdown("#### Major Market Indices")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500", "4,567.89", "2.34%", delta_color="normal")
    with col2:
        st.metric("NASDAQ", "14,123.45", "1.89%", delta_color="normal")
    with col3:
        st.metric("DOW", "35,678.90", "-0.45%", delta_color="inverse")
    with col4:
        st.metric("VIX", "18.45", "-1.23%", delta_color="inverse")

    # Gold prices
    st.markdown("#### 💰 Gold Prices")
    if gold_prices and gold_prices.get('ounce', 0) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gold (1 Ounce)", f"${gold_prices['ounce']:.2f} USD")
        with col2:
            st.metric("Gold (1 Gram)", f"${gold_prices['gram']:.2f} USD")
        with col3:
            st.metric("Gold (1 Kilogram)", f"${gold_prices['kilogram']:.2f} USD")
        
        st.markdown(f"""
        <div class="gold-prices">
            <h3>🌟 Golden Treasure Prices 🌟</h3>
            <p><strong>1 Ounce:</strong> ${gold_prices['ounce']:.2f} USD</p>
            <p><strong>1 Gram:</strong> ${gold_prices['gram']:.2f} USD</p>
            <p><strong>1 Kilogram:</strong> ${gold_prices['kilogram']:.2f} USD</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Gold price data not available. Please ensure API keys are set or check your internet connection.")

def create_technical_analysis_chart(symbol, hist_data):
    """Create technical analysis chart for a stock"""
    if hist_data.empty:
        return None
    
    # Calculate moving averages
    hist_data = hist_data.copy()
    hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
    
    fig = go.Figure()
    
    # Candlestick chart
    if all(col in hist_data.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name=f"{symbol} Price"
        ))
    
    # Moving averages
    if 'MA20' in hist_data.columns:
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['MA20'],
            name='MA20',
            line=dict(color='orange', width=1)
        ))
    
    if 'MA50' in hist_data.columns:
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['MA50'],
            name='MA50',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title=f"{symbol} Technical Analysis",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_risk_gauge(risk_score):
    """Create risk assessment gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Your Risk Score (1-10)"},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3.5], 'color': "lightgreen"},
                {'range': [3.5, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score}
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_portfolio_pie_chart(portfolio_data):
    """Create portfolio allocation pie chart"""
    if not portfolio_data:
        return None
    
    symbols = [item['Stock'] for item in portfolio_data]
    weights = [1/len(portfolio_data)] * len(portfolio_data)  # Equal weights for simplicity
    
    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=weights,
        hole=.3,
        hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        height=400
    )
    
    return fig