"""
Utilities package for Expert Financial Risk Analysis AI.

This package contains utility functions and classes for:
- Advanced financial data processing and analysis
- Interactive visualizations with dark theme
- Document processing for AI vector stores
- Statistical and technical analysis tools

Modules:
    data_processor: Enhanced financial data processing and document conversion
    visualizations: Advanced Plotly visualizations with dark theme support
"""

import logging
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up package logger
logger = logging.getLogger(__name__)

# Import utilities with error handling
try:
    from .data_processor import (
        EnhancedFinancialDataProcessor,
        convert_data_to_documents
    )
    DATA_PROCESSOR_AVAILABLE = True
    logger.info("Data Processor utilities imported successfully")
except ImportError as e:
    logger.warning(f"Could not import Data Processor utilities: {e}")
    DATA_PROCESSOR_AVAILABLE = False
    
    # Create placeholder classes/functions
    class EnhancedFinancialDataProcessor:
        def __init__(self):
            raise ImportError("Data Processor dependencies not available")
    
    def convert_data_to_documents(*args, **kwargs):
        raise ImportError("Data Processor dependencies not available")

try:
    from .visualizations import (
        EnhancedFinancialVisualizations,
        create_dashboard
    )
    VISUALIZATIONS_AVAILABLE = True
    logger.info("Visualizations utilities imported successfully")
except ImportError as e:
    logger.warning(f"Could not import Visualizations utilities: {e}")
    VISUALIZATIONS_AVAILABLE = False
    
    # Create placeholder classes/functions
    class EnhancedFinancialVisualizations:
        def __init__(self):
            raise ImportError("Visualizations dependencies not available")
    
    def create_dashboard(*args, **kwargs):
        raise ImportError("Visualizations dependencies not available")

# Utility availability status
UTILITY_STATUS = {
    'data_processor': DATA_PROCESSOR_AVAILABLE,
    'visualizations': VISUALIZATIONS_AVAILABLE
}

def get_utility_status():
    """Get the availability status of all utilities."""
    return UTILITY_STATUS.copy()

def check_all_utilities():
    """Check if all utilities are available."""
    return all(UTILITY_STATUS.values())

def get_missing_dependencies():
    """Get list of missing dependencies that prevent utilities from loading."""
    missing_deps = []
    
    if not DATA_PROCESSOR_AVAILABLE:
        missing_deps.extend([
            'pandas', 'numpy', 'scikit-learn', 'scipy',
            'langchain-core', 'langchain-text-splitters'
        ])
    
    if not VISUALIZATIONS_AVAILABLE:
        missing_deps.extend(['plotly', 'streamlit'])
    
    return list(set(missing_deps))  # Remove duplicates

def install_missing_dependencies():
    """Provide instructions for installing missing dependencies."""
    missing = get_missing_dependencies()
    if missing:
        deps_str = ' '.join(missing)
        return f"pip install {deps_str}"
    return "All dependencies are satisfied"

# Export main classes and functions
__all__ = [
    # Main utility classes
    'EnhancedFinancialDataProcessor',
    'EnhancedFinancialVisualizations',
    
    # Main utility functions
    'convert_data_to_documents',
    'create_dashboard',
    
    # Status and diagnostic functions
    'get_utility_status',
    'check_all_utilities',
    'get_missing_dependencies',
    'install_missing_dependencies',
    
    # Status variables
    'UTILITY_STATUS',
    'DATA_PROCESSOR_AVAILABLE',
    'VISUALIZATIONS_AVAILABLE'
]

# Package metadata
__version__ = "2.0.0"
__author__ = "Financial AI Team"
__description__ = "Utility functions and classes for Financial AI Assistant"
__package_name__ = "utils"

# Log package initialization
logger.info(f"Utils package initialized. Available utilities: {sum(UTILITY_STATUS.values())}/{len(UTILITY_STATUS)}")
if not check_all_utilities():
    missing_utilities = [name for name, available in UTILITY_STATUS.items() if not available]
    logger.warning(f"Missing utilities: {', '.join(missing_utilities)}")
    missing_deps = get_missing_dependencies()
    if missing_deps:
        logger.info(f"Install missing dependencies with: {install_missing_dependencies()}")
else:
    logger.info("All utilities are available and ready")