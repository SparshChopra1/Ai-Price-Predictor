import requests
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st

class AIPredictor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = config.OPENROUTER_BASE_URL
        self.model = config.OPENROUTER_MODEL
        
    def predict_prices(self, crop: str, state: str, 
                      historical_data: Optional[Dict], 
                      market_data: List[Dict]) -> Dict:
        """Generate AI-powered price predictions"""
        
        # Statistical prediction
        if historical_data and len(historical_data['prices']) > 0:
            prices = np.array(historical_data['prices'])
            
            # Polynomial regression for trend
            X = np.arange(len(prices)).reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, prices)
            
            # Predict next 7 days
            future_X = np.array([[len(prices) + i] for i in range(1, 8)])
            future_X_poly = poly.transform(future_X)
            predictions = model.predict(future_X_poly)
            
            # Add seasonal adjustments
            seasonal_factor = self._get_seasonal_factor(crop)
            predictions *= seasonal_factor
            
            # Calculate metrics
            current_price = float(prices[-1])
            predicted_price = float(predictions[6])  # 7 days ahead
            trend = ((predicted_price - current_price) / current_price) * 100
            
            # Get AI insights
            ai_analysis = self._get_ai_analysis(crop, state, prices, predictions)
            
            # Find best selling day - COMPLETE FIX: Ensure it's a native Python int
            best_day_idx_raw = np.argmax(predictions)
            best_day_idx = int(best_day_idx_raw.item() if hasattr(best_day_idx_raw, 'item') else best_day_idx_raw)
            days_ahead = int(best_day_idx + 1)
            best_day = (datetime.now() + timedelta(days=days_ahead)).strftime('%A, %d %B')
            
            expected_profit_value = float(predictions[best_day_idx]) - current_price
            
            return {
                'predicted_price': round(predicted_price, 0),
                'trend': round(float(trend), 1),
                'confidence': self._calculate_confidence(prices, model.score(X_poly, prices)),
                'recommendation': self._generate_recommendation(current_price, predicted_price, trend),
                'best_day': best_day,
                'expected_profit': round(expected_profit_value, 0),
                'factors': ai_analysis.get('factors', self._default_factors()),
                'daily_predictions': [float(p) for p in predictions.tolist()]
            }
        
        # Fallback prediction
        return self._generate_fallback_prediction(crop, state)
    
    def _get_seasonal_factor(self, crop: str) -> float:
        """Calculate seasonal adjustment factor"""
        month = datetime.now().month
        
        seasonal_patterns = {
            'Wheat': {3: 0.95, 4: 0.90, 10: 1.05, 11: 1.08, 12: 1.10},
            'Rice': {10: 0.92, 11: 0.90, 3: 1.05, 4: 1.08},
            'Onion': {1: 1.15, 2: 1.20, 6: 0.85, 7: 0.80},
            'Cotton': {10: 0.95, 11: 0.90, 3: 1.10, 4: 1.15},
            'Potato': {2: 0.90, 3: 0.85, 9: 1.10, 10: 1.15},
            'Tomato': {4: 0.95, 5: 0.90, 11: 1.05, 12: 1.10},
            'Maize': {3: 0.93, 4: 0.90, 10: 1.05, 11: 1.08},
            'Soyabean': {10: 0.92, 11: 0.88, 3: 1.08, 4: 1.12}
        }
        
        if crop in seasonal_patterns and month in seasonal_patterns[crop]:
            return seasonal_patterns[crop][month]
        return 1.0
    
    def _get_ai_analysis(self, crop: str, state: str, 
                        historical_prices: np.ndarray, 
                        predictions: np.ndarray) -> Dict:
        """Get AI analysis using OpenRouter API"""
        try:
            # Convert numpy values to regular Python types
            recent_prices = [float(p) for p in historical_prices[-5:].tolist()]
            avg_prediction = float(predictions.mean())
            
            prompt = f"""Analyze market conditions for {crop} in {state}:
Recent prices (last 5 days): {recent_prices}
Predicted next week average: ₹{avg_prediction:.0f}/qtl

Provide brief JSON response with these exact keys:
{{"factors": {{"demand": "High/Medium/Low", "supply": "Surplus/Normal/Shortage", "market": "Bullish/Neutral/Bearish", "weather": "Favorable/Challenging"}}}}

Keep it concise and use only these exact values."""
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://ai-crop-doctor.com',
                'X-Title': 'AI Market Predictor'
            }
            
            data = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': 'You are an agricultural market analyst. Respond only with valid JSON.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 200
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                # Try to parse JSON from response
                try:
                    # Remove markdown code blocks if present
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()
                    
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError:
                    print(f"JSON Parse Error: {content}")
                    return {'factors': self._default_factors()}
                    
        except Exception as e:
            print(f"AI API Error: {str(e)}")
        
        return {'factors': self._default_factors()}
    
    def _default_factors(self) -> Dict:
        """Default market factors"""
        return {
            'demand': np.random.choice(['High', 'Medium', 'Low']),
            'supply': np.random.choice(['Surplus', 'Normal', 'Shortage']),
            'market': np.random.choice(['Bullish', 'Neutral', 'Bearish']),
            'weather': np.random.choice(['Favorable', 'Challenging'])
        }
    
    def _calculate_confidence(self, prices: np.ndarray, model_score: float) -> int:
        """Calculate prediction confidence"""
        base_confidence = model_score * 100
        volatility = float(np.std(prices)) / float(np.mean(prices))
        volatility_factor = max(0.7, 1 - volatility)
        confidence = base_confidence * volatility_factor
        return int(min(95, max(60, confidence)))
    
    def _generate_recommendation(self, current: float, predicted: float, trend: float) -> str:
        """Generate selling recommendation"""
        trend_val = float(trend)
        current_val = float(current)
        predicted_val = float(predicted)
        
        if trend_val > 5:
            return f"Prices expected to rise by {abs(trend_val):.1f}%. Consider holding your stock for better returns. Potential gain of ₹{predicted_val - current_val:.0f}/qtl."
        elif trend_val < -5:
            return f"Prices may decline by {abs(trend_val):.1f}%. Consider selling soon to avoid potential losses of ₹{abs(predicted_val - current_val):.0f}/qtl."
        else:
            return f"Market is stable with {abs(trend_val):.1f}% expected change. Current price of ₹{current_val:.0f}/qtl is fair. You can sell based on your needs."
    
    def _generate_fallback_prediction(self, crop: str, state: str) -> Dict:
        """Generate fallback prediction when data is limited"""
        base_prices = {
            'Wheat': 2150, 'Rice': 2800, 'Cotton': 6500, 'Maize': 1850,
            'Potato': 1200, 'Onion': 1800, 'Tomato': 2500, 'Soyabean': 4200
        }
        
        base = base_prices.get(crop, 2000)
        trend = float(np.random.uniform(-5, 5))
        predicted = base * (1 + trend/100)
        
        # Ensure days parameter is native Python int
        best_days_ahead = 3
        best_day = (datetime.now() + timedelta(days=best_days_ahead)).strftime('%A, %d %B')
        
        return {
            'predicted_price': round(predicted, 0),
            'trend': round(trend, 1),
            'confidence': 75,
            'recommendation': f"Market conditions are stable. Current price around ₹{base}/qtl. Monitor daily prices for best selling opportunity.",
            'best_day': best_day,
            'expected_profit': round(abs(trend) * base / 100, 0),
            'factors': self._default_factors(),
            'daily_predictions': [float(base * (1 + i*0.005)) for i in range(7)]
        }
    
    def get_smart_advice(self, crop: str, state: str, current_price: float) -> Dict:
        """Generate personalized selling advice"""
        try:
            price_val = float(current_price)
            
            prompt = f"""Generate practical selling advice for {crop} farmers in {state}.
Current market price: ₹{price_val:.0f}/quintal

Provide actionable advice considering:
- Storage costs (₹50-100/qtl per month)
- Quality degradation (2-3% per month)
- Market timing and demand cycles
- Transportation and handling costs

Format as JSON:
{{"main_advice": "50 words main advice", "point1": "20 words action point 1", "point2": "20 words action point 2", "point3": "20 words action point 3"}}"""
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://ai-crop-doctor.com',
                'X-Title': 'AI Market Advisor'
            }
            
            data = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': 'You are an expert agricultural market advisor helping Indian farmers maximize profits. Respond only with valid JSON.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.8,
                'max_tokens': 400
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                try:
                    # Remove markdown code blocks if present
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()
                    
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError:
                    print(f"Advice JSON Parse Error: {content}")
                    pass
                    
        except Exception as e:
            print(f"Advice API Error: {str(e)}")
        
        # Fallback advice
        price_val = float(current_price)
        return {
            'main_advice': f"Current price of ₹{price_val:.0f}/qtl is {'good' if price_val > 2000 else 'moderate'}. Consider selling 60-70% of your stock now to secure profits, and hold the remaining 30-40% for potential price improvements in the next 7-10 days.",
            'point1': "Check moisture content and ensure it's within acceptable limits (12-14% for grains) to avoid price deductions at mandi.",
            'point2': "Compare prices across 3-4 nearby mandis before selling. Price differences of ₹50-200/qtl are common between markets.",
            'point3': "Factor in storage costs of ₹50-100/qtl per month and quality degradation of 2-3% when deciding to hold stock."
        }
    
    def get_top_crops(self, state: str) -> List[Dict]:
        """Get top value crops for the region"""
        state_specific_crops = {
            'Punjab': [
                {'name': 'Basmati Rice', 'price': 3800},
                {'name': 'Wheat (Sharbati)', 'price': 2400},
                {'name': 'Cotton', 'price': 6500}
            ],
            'Haryana': [
                {'name': 'Basmati Rice', 'price': 3700},
                {'name': 'Wheat', 'price': 2300},
                {'name': 'Mustard', 'price': 5200}
            ],
            'Maharashtra': [
                {'name': 'Cotton (Premium)', 'price': 6800},
                {'name': 'Soyabean', 'price': 4500},
                {'name': 'Sugarcane', 'price': 3200}
            ],
            'Karnataka': [
                {'name': 'Coffee', 'price': 8500},
                {'name': 'Cotton', 'price': 6400},
                {'name': 'Maize', 'price': 1900}
            ]
        }
        
        crops = state_specific_crops.get(state, [
            {'name': 'Basmati Rice', 'price': 3500},
            {'name': 'Cotton (Premium)', 'price': 6800},
            {'name': 'Soyabean', 'price': 4500}
        ])
        
        return sorted(crops, key=lambda x: x['price'], reverse=True)[:3]
    
    def get_seasonal_forecast(self, crop: str) -> Dict:
        """Get seasonal demand forecast"""
        month = datetime.now().month
        season = 'Rabi' if month in [10, 11, 12, 1, 2, 3] else 'Kharif'
        
        forecasts = {
            'Wheat': {
                'trend': 'rising' if month in [1, 2, 8, 9] else 'stable',
                'summary': f"Wheat demand typically increases during festive season and pre-harvest period. Export demand remains strong. Current {season} season may see price stability."
            },
            'Rice': {
                'trend': 'high' if month in [8, 9, 10, 11] else 'moderate',
                'summary': f"Rice sees consistent demand year-round. {season} harvest may affect prices. Government procurement provides price support floor."
            },
            'Cotton': {
                'trend': 'rising' if month in [10, 11, 12, 1] else 'stable',
                'summary': f"Cotton prices influenced by textile industry demand and global markets. Peak harvest season (Oct-Jan) may see price pressure."
            },
            'Onion': {
                'trend': 'high' if month in [1, 2, 6, 7] else 'moderate',
                'summary': f"Onion prices highly volatile. Summer shortage (May-July) and winter demand (Dec-Feb) drive prices up. Storage plays key role."
            },
            'Potato': {
                'trend': 'rising' if month in [6, 7, 8, 9] else 'stable',
                'summary': f"Potato prices peak during monsoon months when fresh supply is limited. Cold storage availability affects pricing significantly."
            },
            'Tomato': {
                'trend': 'high' if month in [4, 5, 6, 7] else 'moderate',
                'summary': f"Tomato prices peak during summer due to reduced production. Highly perishable nature leads to price volatility."
            },
            'Maize': {
                'trend': 'rising' if month in [2, 3, 8, 9] else 'stable',
                'summary': f"Maize demand driven by poultry and starch industries. {season} production levels will influence upcoming prices."
            },
            'Soyabean': {
                'trend': 'high' if month in [10, 11, 12] else 'moderate',
                'summary': f"Soyabean prices linked to global edible oil markets. Post-harvest (Oct-Dec) sees price discovery. Export potential impacts rates."
            }
        }
        
        if crop in forecasts:
            return forecasts[crop]
        
        return {
            'trend': 'stable',
            'summary': f"{crop} demand follows seasonal consumption patterns. Monitor local market trends and government procurement announcements for price signals."
        }