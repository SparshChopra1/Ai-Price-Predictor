import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
DATA_GOV_API_KEY = os.getenv('579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b', '')
OPENROUTER_API_KEY = os.getenv('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', '')

# Data.gov.in API Configuration
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource/"
MARKET_PRICE_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# OpenRouter Configuration  
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "anthropic/claude-3-haiku"  # Fast and cost-effective

# Application Settings
SUPPORTED_STATES = [
    "Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh",
    "Maharashtra", "Karnataka", "Rajasthan", "Gujarat",
    "West Bengal", "Bihar", "Andhra Pradesh", "Tamil Nadu"
]

SUPPORTED_CROPS = [
    "Wheat", "Rice", "Cotton", "Maize", "Potato", 
    "Onion", "Tomato", "Soyabean", "Groundnut", "Arhar"
]

# Cache Settings
CACHE_TTL = 1800  # 30 minutes

# Theme Colors
COLORS = {
    'primary_green': '#4CAF50',
    'light_beige': '#F9F9F6',
    'sky_blue': '#2196F3',
    'accent_orange': '#FF9800',
    'white': '#FFFFFF',
    'text_dark': '#2C3E50',
    'text_light': '#7F8C8D'
}