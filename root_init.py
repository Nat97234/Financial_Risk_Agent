"""
Expert Financial Risk Analysis AI Platform

A comprehensive AI-powered financial analysis platform that provides:

🤖 **AI-Powered Analysis**
- GPT-4 powered financial agent with expert knowledge
- Real-time web research capabilities
- Personalized investment recommendations
- Multi-language support (English/Arabic)

📊 **Real-Time Market Data**
- Live stock prices and analysis
- Cryptocurrency market tracking
- Commodities and precious metals (Gold focus)
- Forex market analysis
- Economic indicators and sentiment

📈 **Advanced Analytics**
- Technical analysis with interactive charts
- Risk assessment and portfolio optimization
- Correlation analysis across asset classes
- Value at Risk (VaR) calculations
- Fundamental analysis and valuation

🎨 **Professional Interface**
- Enhanced dark theme
- Interactive Plotly visualizations
- Real-time dashboard updates
- Mobile-responsive design

💡 **Key Features**
- Automated user profiling from conversations
- Multi-source data integration (Yahoo Finance, World Bank, Alpha Vantage)
- Professional financial calculations and metrics
- Comprehensive market sentiment analysis
- Export and portfolio tracking capabilities

⚠️ **Disclaimer**
This platform is for educational and informational purposes only.
Always consult qualified financial advisors for investment decisions.

Author: Financial AI Team
Version: 2.0.0
License: MIT
"""

import sys
import logging
from pathlib import Path

# Set up root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "2.0.0"
__author__ = "Financial AI Team"
__email__ = "support@financialai.com"
__description__ = "Expert Financial Risk Analysis AI Platform"
__url__ = "https://github.com/your-repo/financial-ai-assistant"
__license__ = "MIT"
__status__ = "Production"

# Python version check
REQUIRED_PYTHON = (3, 8)
if sys.version_info < REQUIRED_PYTHON:
    raise RuntimeError(f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ is required")

# Import main components with error handling
try:
    # Import configuration
    from config import APP_NAME, VERSION, DEBUG
    CONFIG_AVAILABLE = True
    logger.info("Configuration loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load configuration: {e}")
    CONFIG_AVAILABLE = False
    APP_NAME = "Expert Financial Risk Analysis AI"
    VERSION = "2.0.0"
    DEBUG = False

try:
    # Import services
    from services import (
        EnhancedFinancialAIAgent,
        EnhancedDataFetcher,
        EnhancedUserProfileManager,
        get_service_status,
        check_all_services
    )
    SERVICES_AVAILABLE = True
    logger.info("Services loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load services: {e}")
    SERVICES_AVAILABLE = False

try:
    # Import utilities
    from utils import (
        EnhancedFinancialDataProcessor,
        EnhancedFinancialVisualizations,
        get_utility_status,
        check_all_utilities
    )
    UTILS_AVAILABLE = True
    logger.info("Utilities loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load utilities: {e}")
    UTILS_AVAILABLE = False

# System status
SYSTEM_STATUS = {
    'config': CONFIG_AVAILABLE,
    'services': SERVICES_AVAILABLE,
    'utils': UTILS_AVAILABLE
}

def get_system_status():
    """Get the overall system status."""
    status = SYSTEM_STATUS.copy()
    if SERVICES_AVAILABLE:
        try:
            status['service_details'] = get_service_status()
        except:
            status['service_details'] = {}
    
    if UTILS_AVAILABLE:
        try:
            status['utility_details'] = get_utility_status()
        except:
            status['utility_details'] = {}
    
    return status

def check_system_health():
    """Check overall system health."""
    return all(SYSTEM_STATUS.values())

def get_system_info():
    """Get comprehensive system information."""
    return {
        'app_name': APP_NAME,
        'version': __version__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'status': get_system_status(),
        'healthy': check_system_health(),
        'debug_mode': DEBUG
    }

def print_startup_banner():
    """Print startup banner with system information."""
    banner = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                     {APP_NAME}                     ║
║                                Version {__version__}                                ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  🤖 AI-Powered Financial Analysis Platform                                    ║
║  📊 Real-time Market Data & Analytics                                         ║
║  📈 Professional Investment Tools                                             ║
║  🎨 Enhanced Dark Theme Interface                                             ║
║                                                                                ║
║  Status: {'🟢 HEALTHY' if check_system_health() else '🔴 ISSUES DETECTED'}                                                      ║
║  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}                                                            ║
║  Debug:  {'ON' if DEBUG else 'OFF'}                                                                ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Print any warnings
    if not check_system_health():
        print("⚠️  SYSTEM WARNINGS:")
        for component, status in SYSTEM_STATUS.items():
            if not status:
                print(f"   ❌ {component.upper()}: Not available")
        print()

# Export main components
__all__ = [
    # Metadata
    '__version__',
    '__author__',
    '__description__',
    'APP_NAME',
    'VERSION',
    
    # Main classes (if available)
    'EnhancedFinancialAIAgent',
    'EnhancedDataFetcher', 
    'EnhancedUserProfileManager',
    'EnhancedFinancialDataProcessor',
    'EnhancedFinancialVisualizations',
    
    # System functions
    'get_system_status',
    'check_system_health',
    'get_system_info',
    'print_startup_banner',
    
    # Status variables
    'SYSTEM_STATUS',
    'CONFIG_AVAILABLE',
    'SERVICES_AVAILABLE',
    'UTILS_AVAILABLE'
]

# Initialize logging for the package
logger.info(f"{APP_NAME} v{__version__} package initialized")
if not check_system_health():
    logger.warning("Some components are not available. Check system status for details.")