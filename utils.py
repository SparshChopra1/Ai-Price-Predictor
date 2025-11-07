import numpy as np
from typing import List, Dict

def format_price(price: float) -> str:
    """Format price with Indian numbering"""
    return f"₹{price:,.0f}"

def calculate_trend(old_price: float, new_price: float) -> float:
    """Calculate percentage change"""
    if old_price == 0:
        return 0
    return ((new_price - old_price) / old_price) * 100

def get_price_color(price: float, all_prices: List[Dict]) -> str:
    """Determine price level color"""
    if not all_prices:
        return 'price-average'
    
    prices = [p['modal_price'] for p in all_prices]
    avg = np.mean(prices)
    
    if price > avg * 1.05:
        return 'price-high'
    elif price < avg * 0.95:
        return 'price-low'
    else:
        return 'price-average'

def generate_mock_prices(base: float, days: int) -> List[float]:
    """Generate realistic price series"""
    prices = []
    current = base
    
    for _ in range(days):
        change = np.random.normal(0, base * 0.02)  # 2% std deviation
        current = max(base * 0.8, min(base * 1.2, current + change))
        prices.append(round(current, 0))
    
    return prices