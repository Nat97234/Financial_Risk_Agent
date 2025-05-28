# 🤖 Financial AI Assistant

An advanced AI-powered financial analysis platform built with Streamlit, LangChain, and AWS.

## 🌟 Features

- **Agentic AI System**: GPT-4 powered financial agent with web research capabilities
- **Multi-Source Data Integration**: Yahoo Finance, World Bank, Alpha Vantage, and more
- **Real-time Analysis**: Live market data and sentiment analysis
- **Personalized Insights**: User profile-based recommendations
- **Interactive Dashboards**: Advanced visualizations with Plotly
- **Technical Analysis**: Candlestick charts, moving averages, risk assessment
- **Scalable Architecture**: AWS ECS with auto-scaling and load balancing

## 🏗️ Architecture

```
Frontend (Streamlit) → AI Agent (LangChain) → Multiple Data Sources
                    ↓
               AWS Infrastructure
              (ECS, ALB, ECR, Secrets)
```

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS CLI (for deployment)
- OpenAI API Key

### Local Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd financial-ai-assistant
```

2. **Create project structure**

```bash
mkdir -p config services utils static aws
touch config/__init__.py services/__init__.py utils/__init__.py
```

3. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Environment Configuration**

```bash
cp .env.template .env
# Edit .env with your API keys
```

5. **Add your CSV file**

```bash
# Place financial_risk_analysis_large.csv in the root directory
```

6. **Run locally**

```bash
streamlit run app.py
```

7. **Or use Docker Compose**

```bash
docker-compose up --build
```

Access the application at `http://localhost:8501`

## 📁 Project Structure

```
financial-ai-assistant/
├── app.py                      # Main Streamlit application
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration and environment variables
├── services/
│   ├── __init__.py
│   ├── data_fetcher.py        # Data fetching services
│   ├── ai_agent.py            # AI agent and tools
│   └── user_profile.py        # User profile management
├── utils/
│   ├── __init__.py
│   ├── data_processor.py      # Data processing utilities
│   └── visualizations.py     # Chart and visualization functions
├── static/
│   └── styles.css             # CSS styles
├── financial_risk_analysis_large.csv  # Your CSV data file
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not in repo)
├── .gitignore                 # Git ignore file
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker compose for local development
└── README.md                  # Project documentation
```

## 🔧 Configuration

### Environment Variables

| Variable               | Description                     | Required |
| ---------------------- | ------------------------------- | -------- |
| `OPENAI_API_KEY`       | OpenAI API key for AI agent     | Yes      |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage API key           | No       |
| `FMP_API_KEY`          | Financial Modeling Prep API key | No       |
| `GOLDAP_API`           | Gold API key                    | No       |
| `DEBUG`                | Enable debug mode               | No       |

### API Keys Setup

1. **OpenAI API** (Required)

   - Sign up at https://openai.com/api/
   - Generate API key
   - Add to `.env` file

2. **Alpha Vantage** (Optional)

   - Sign up at https://www.alphavantage.co/
   - Get free API key
   - Provides additional stock data

3. **Other APIs** (Optional)
   - Financial Modeling Prep: https://financialmodelingprep.com/
   - Gold API: https://www.goldapi.io/
   - Each provides enhanced data features

## 📊 Usage

### Basic Usage

1. **Ask Questions**: Use natural language to ask about stocks, investments, or market analysis
2. **View Dashboards**: Explore interactive charts and real-time data
3. **Get Personalized Advice**: The AI learns your preferences and provides tailored recommendations

### Example Queries

- "Should I invest in Tesla stock given my moderate risk tolerance?"
- "What's the current market sentiment for tech stocks?"
- "Compare Apple and Microsoft performance over the last 6 months"
- "What are the best investment strategies for a 30-year-old?"

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**

   - Verify API keys in `.env` file
   - Check API key quotas and limits

2. **Import Errors**

   - Make sure all `__init__.py` files are created
   - Check Python path and virtual environment

3. **Data Loading Issues**

   - Ensure `financial_risk_analysis_large.csv` is in root directory
   - Check file permissions and format

4. **Performance Issues**
   - Monitor memory usage with large CSV files
   - Consider data sampling for better performance

## 🧪 Testing

### Local Testing

```bash
# Test basic functionality
python -c "from services.data_fetcher import DataFetcher; print('Import successful')"

# Test Docker build
docker build -t financial-ai-assistant-test .
docker run -p 8501:8501 financial-ai-assistant-test
```

## 📝 Development

### Adding New Features

1. **New Data Sources**

   - Add fetching logic to `services/data_fetcher.py`
   - Update configuration in `config/settings.py`

2. **New Visualizations**

   - Add chart functions to `utils/visualizations.py`
   - Update dashboard in main app

3. **New AI Tools**
   - Extend `services/ai_agent.py`
   - Add new tool functions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For support and questions:

- Create an issue in the GitHub repository
- Check the troubleshooting section above

---

**Happy Trading! 📈🤖**
