import streamlit as st
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
from config.settings import *


class FinancialAgentTools:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore
        try:
            self.search_tool = DuckDuckGoSearchRun()
        except Exception as e:
            st.warning(f"Web search tool initialization failed: {e}")
            self.search_tool = None
    
    def financial_data_search(self, query: str) -> str:
        """Search internal financial database"""
        if not self.vectorstore:
            return "Internal financial database not available"
        
        try:
            docs = self.vectorstore.similarity_search(query, k=5)
            results = "\n\n".join([doc.page_content for doc in docs])
            return f"Internal Financial Data Results:\n{results}"
        except Exception as e:
            return f"Error searching financial data: {str(e)}"
    
    def web_research(self, query: str) -> str:
        """Search the web for financial information"""
        if not self.search_tool:
            return "Web research tool not available"
        
        try:
            financial_query = f"financial market analysis {query} 2024 2025"
            results = self.search_tool.run(financial_query)
            return f"Web Research Results:\n{results}"
        except Exception as e:
            return f"Web research unavailable: {str(e)}"
    
    def calculate_financial_metrics(self, data: str) -> str:
        """Calculate financial metrics and ratios"""
        try:
            data_lower = data.lower()
            if "p/e ratio" in data_lower or "pe ratio" in data_lower:
                return "P/E Ratio calculation: Price per Share / Earnings per Share. Lower ratios may indicate undervalued stocks, while higher ratios might suggest growth expectations."
            elif "roi" in data_lower or "return on investment" in data_lower:
                return "ROI calculation: (Gain from Investment - Cost of Investment) / Cost of Investment × 100%"
            elif "market cap" in data_lower:
                return "Market Cap calculation: Current Stock Price × Total Outstanding Shares"
            elif "dividend yield" in data_lower:
                return "Dividend Yield calculation: Annual Dividends per Share / Price per Share × 100%"
            else:
                return "Financial metrics can include P/E ratio, ROI, Market Cap, Dividend Yield, Beta, and more. Please specify which metric you'd like to calculate."
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def get_market_sentiment(self, query: str) -> str:
        """Analyze market sentiment"""
        if not self.search_tool:
            return "Market sentiment analysis requires web search capability"
        
        try:
            sentiment_query = f"market sentiment analysis {query} investor opinion 2024"
            results = self.search_tool.run(sentiment_query)
            return f"Market Sentiment Analysis:\n{results}"
        except Exception as e:
            return f"Sentiment analysis unavailable: {str(e)}"


class FinancialAIAgent:
    def __init__(self):
        self.vectorstore = None
        self.agent_executor = None
        self.memory = ConversationBufferWindowMemory(
            k=10, 
            return_messages=True,
            memory_key="chat_history"
        )
        if OPENAI_API_KEY:
            self.initialize_agent()
    
    def create_vectorstore(self, documents):
        """Create vector store from documents"""
        if not documents or not OPENAI_API_KEY:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        try:
            if os.path.exists("faiss_index"):
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = FAISS.load_local(
                    "faiss_index", 
                    embeddings=embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                vectorstore.save_local("faiss_index")
            return vectorstore
        except Exception as e:
            st.warning(f"Vector store creation failed: {e}")
            return None
    
    def initialize_agent(self):
        """Initialize the AI agent with tools"""
        if not OPENAI_API_KEY:
            st.error("OpenAI API key is required for AI agent functionality")
            return
        
        try:
            llm = ChatOpenAI(
                model_name="gpt-4", 
                temperature=0.1, 
                openai_api_key=OPENAI_API_KEY
            )
            
            financial_tools = FinancialAgentTools(self.vectorstore)
            
            tools = [
                Tool(
                    name="FinancialDataSearch",
                    func=financial_tools.financial_data_search,
                    description="Search internal financial database for stock prices, company info, economic indicators"
                ),
                Tool(
                    name="WebResearch",
                    func=financial_tools.web_research,
                    description="Search the web for current financial news, market analysis, and additional information"
                ),
                Tool(
                    name="CalculateMetrics",
                    func=financial_tools.calculate_financial_metrics,
                    description="Calculate financial ratios, metrics, and perform quantitative analysis"
                ),
                Tool(
                    name="MarketSentiment",
                    func=financial_tools.get_market_sentiment,
                    description="Analyze market sentiment and investor opinions"
                )
            ]
            
            # Use updated prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an Advanced Financial AI Agent with access to multiple data sources and web research capabilities.

Your capabilities include:
1. Analyzing internal financial databases (stocks, economics, company data)
2. Conducting web research for the latest information when internal data is insufficient
3. Calculating financial metrics and ratios
4. Analyzing market sentiment
5. Providing personalized financial insights

When answering questions:
1. Always start by acknowledging the user's question
2. First, use the FinancialDataSearch tool to check internal data
3. If internal data is insufficient, automatically use the WebResearch tool
4. Use CalculateMetrics for financial calculations
5. Use MarketSentiment for sentiment analysis
6. Combine multiple data sources for complete analysis
7. Present information in an organized, detailed manner
8. Provide actionable insights and recommendations

Remember: Always be thorough, accurate, and helpful."""),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            
            try:
                react_prompt = hub.pull("hwchase17/react")
            except Exception:
                # Fallback prompt if hub is unavailable
                react_prompt = prompt_template
            
            agent = create_react_agent(llm, tools, react_prompt)
            self.agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True, 
                max_iterations=5,
                memory=self.memory,
                handle_parsing_errors=True
            )
            
        except Exception as e:
            st.error(f"Agent initialization failed: {e}")
            self.agent_executor = None
    
    def update_vectorstore(self, documents):
        """Update the vector store with new documents"""
        self.vectorstore = self.create_vectorstore(documents)
        if self.agent_executor and self.vectorstore:
            # Update the financial tools with new vectorstore
            for tool in self.agent_executor.tools:
                if tool.name == "FinancialDataSearch":
                    tool.func = FinancialAgentTools(self.vectorstore).financial_data_search
    
    def process_question(self, question: str) -> str:
        """Process a question using the AI agent"""
        if not self.agent_executor:
            return "AI Agent is not available. Please check your OpenAI API key and try again."
        
        try:
            response = self.agent_executor.invoke({"input": question})
            answer = response.get("output", "I apologize, but I couldn't generate a response.")
            
            # Update memory
            self.memory.chat_memory.add_user_message(HumanMessage(content=question))
            self.memory.chat_memory.add_ai_message(AIMessage(content=answer))
            
            return answer
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            st.error(error_msg)
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing it or check if your API keys are properly configured."
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()