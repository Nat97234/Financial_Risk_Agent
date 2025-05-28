import streamlit as st
import pandas as pd
import datetime
import uuid
from config.settings import *
from services.data_fetcher import DataFetcher
from services.ai_agent import FinancialAIAgent
from services.user_profile import UserProfileManager
from utils.visualizations import create_dashboard
from utils.data_processor import convert_data_to_documents

# Set page config
st.set_page_config(
    page_title="🤖 Agentic AI Financial Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Load CSS
try:
    with open('static/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found. Using default styling.")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = FinancialAIAgent()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Agentic AI Financial Assistant</h1>
        <p>Empowering Wealth Creation with Real-time Insights & Personalized Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize services
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher()
    
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserProfileManager()

    # Fetch data
    with st.spinner("🔄 Loading financial data from multiple sources..."):
        combined_data = st.session_state.data_fetcher.fetch_all_data()
        
        # Update AI agent with data if not already done
        if not st.session_state.data_loaded:
            documents = convert_data_to_documents(combined_data)
            if documents:
                st.session_state.ai_agent.update_vectorstore(documents)
            st.session_state.data_loaded = True

    # Display data sources status
    display_data_sources_status(combined_data)

    # Main chat interface
    chat_interface()

    # Dashboard
    create_dashboard(combined_data)

    # Sidebar
    create_sidebar()

def display_data_sources_status(combined_data):
    st.markdown('<h2 class="section-header">📊 Data Sources Status</h2>', unsafe_allow_html=True)
    
    sources_status = [
        ("Yahoo Finance", len(combined_data.get('yfinance_data', {})), "active"),
        ("World Bank", len(combined_data.get('world_bank_data', {})), "active" if combined_data.get('world_bank_data') else "inactive"),
        ("Alpha Vantage", len(combined_data.get('alpha_data', {})), "active" if ALPHA_API_KEY else "limited"),
        ("Web Research", "Available", "active")
    ]

    cols = st.columns(4)
    for i, (source, count, status) in enumerate(sources_status):
        with cols[i]:
            st.markdown(f"""
            <div class="data-source-card">
                <div class="data-source-info">
                    <span class="status-dot status-{status}"></span>
                    <div>
                        <strong>{source}</strong>
                        <br>
                        <small>{count}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def chat_interface():
    st.markdown('<h2 class="section-header">💬 Ask Your Financial AI Agent</h2>', unsafe_allow_html=True)

    if not OPENAI_API_KEY:
        st.error("🔑 Please set your OPENAI_API_KEY in the .env file to use the Agentic AI features.")
        st.info("Add your OpenAI API key to the .env file: OPENAI_API_KEY=your_key_here")
        return

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_area(
            "Ask me anything about finance, markets, investments, or economics:", 
            placeholder="e.g., Should I invest in Tesla given my moderate risk tolerance and 10-year timeline?",
            height=100,
            key="question_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button("🚀 Ask Agent", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)

    if clear_button:
        st.session_state.conversation_history = []
        st.session_state.ai_agent.clear_memory()
        st.success("Chat history cleared!")
        st.rerun()

    if question and submit_button:
        handle_question(question)

def handle_question(question):
    st.markdown(f"""
    <div class="question-box">
        <h3>❓ Your Question:</h3>
        <p style="font-size: 1.1em;">{question}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Collect user info if needed
    user_context = st.session_state.user_manager.collect_and_process_user_info(question)
    enhanced_question = f"{question}\n{user_context}"
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.markdown('<div class="agent-thinking">🔍 Analyzing your question...</div>', unsafe_allow_html=True)
        progress_bar.progress(25)
        
        status_text.markdown('<div class="agent-thinking">📊 Searching financial databases...</div>', unsafe_allow_html=True)
        progress_bar.progress(50)
        
        status_text.markdown('<div class="agent-thinking">🌐 Conducting web research if needed...</div>', unsafe_allow_html=True)
        progress_bar.progress(75)
        
        status_text.markdown('<div class="agent-thinking">🧠 Generating comprehensive analysis...</div>', unsafe_allow_html=True)
        progress_bar.progress(90)
        
        # Process with AI agent
        response = st.session_state.ai_agent.process_question(enhanced_question)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.markdown(f"""
        <div class="answer-box">
            <h3>🤖 AI Agent Response:</h3>
            <div style="font-size: 1.05em; line-height: 1.6;">{response}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Save to conversation history
        st.session_state.conversation_history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": response,
            "user_context": user_context
        })
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error processing your question: {e}")
        st.info("Please try rephrasing your question or check your API configuration.")

def create_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h3>🤖 Agent Status</h3></div>', unsafe_allow_html=True)
        
        if OPENAI_API_KEY and st.session_state.ai_agent.agent_executor:
            st.success("✅ Agentic AI Active")
            st.info("🧠 GPT-4 Powered")
            st.info("🔍 Web Research Enabled")
            st.info("📊 Multi-Source Analysis")
        else:
            st.error("❌ Agent Offline")
            if not OPENAI_API_KEY:
                st.warning("Missing OpenAI API Key")
        
        # User profile section
        st.markdown('<div class="sidebar-section"><h3>👤 Your Profile</h3></div>', unsafe_allow_html=True)
        if st.session_state.user_profile:
            for key, value in st.session_state.user_profile.items():
                st.write(f"**{key.title()}**: {value}")
            
            if st.button("🔄 Clear Profile"):
                st.session_state.user_profile = {}
                st.rerun()
        else:
            st.info("Profile will be built as you ask personalized questions")
        
        # Conversation history
        st.markdown('<div class="sidebar-section"><h3>💬 Recent Conversations</h3></div>', unsafe_allow_html=True)
        if st.session_state.conversation_history:
            for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {entry['question'][:40]}..."):
                    st.write(f"**Time**: {entry['timestamp']}")
                    st.write(f"**Question**: {entry['question']}")
                    st.write(f"**Answer**: {entry['answer'][:200]}...")
        else:
            st.info("No conversations yet")
        
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.ai_agent.clear_memory()
            st.success("History cleared!")
            st.rerun()

        # Agent Tools Info
        st.markdown('<div class="sidebar-section"><h3>🛠️ Agent Tools</h3></div>', unsafe_allow_html=True)
        st.write("✅ Financial Database Search")
        st.write("✅ Real-time Web Research")  
        st.write("✅ Market Sentiment Analysis")
        st.write("✅ Financial Calculations")
        st.write("✅ Personalized Recommendations")
        
        st.markdown('<div class="sidebar-section"><h3>🎯 Agent Features</h3></div>', unsafe_allow_html=True)
        st.write("🧠 Memory of conversations")
        st.write("👤 User profile building")
        st.write("🔍 Multi-step reasoning")
        st.write("📊 Data visualization")
        st.write("🌐 Internet research")
        st.write("📈 Real-time analysis")

if __name__ == "__main__":
    main()