import requests
from typing import Dict, List
import streamlit as st

class LocationService:
    def __init__(self):
        # Real districts for each state
        self.states_districts = {
            'Punjab': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda', 'Mohali', 'Jagraon', 'Moga'],
            'Haryana': ['Gurugram', 'Faridabad', 'Karnal', 'Rohtak', 'Hisar', 'Panipat', 'Ambala', 'Sonipat'],
            'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi', 'Meerut', 'Ghaziabad', 'Allahabad', 'Bareilly'],
            'Madhya Pradesh': ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior', 'Ujjain', 'Sagar', 'Dewas', 'Satna'],
            'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad', 'Solapur', 'Kolhapur', 'Ahmednagar'],
            'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum', 'Davangere', 'Gulbarga', 'Shimoga'],
            'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer', 'Bikaner', 'Alwar', 'Bharatpur'],
            'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Gandhinagar', 'Bhavnagar', 'Jamnagar', 'Anand'],
            'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Siliguri', 'Asansol', 'Bardhaman', 'Malda', 'Hooghly'],
            'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Darbhanga', 'Purnia', 'Arrah', 'Bihar Sharif'],
            'Andhra Pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool', 'Tirupati', 'Anantapur', 'Rajahmundry'],
            'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 'Tirunelveli', 'Erode', 'Vellore']
        }
    
    def get_user_location(self) -> Dict:
        """Get user location from IP"""
        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'city': data.get('city', 'Unknown'),
                    'state': data.get('region', 'Unknown'),
                    'country': data.get('country_name', 'India')
                }
        except:
            pass
        
        return {'city': 'New Delhi', 'state': 'Delhi', 'country': 'India'}
    
    def get_nearby_districts(self, state: str) -> List[str]:
        """Get districts in a state"""
        return self.states_districts.get(state, ['Central District', 'North District', 'South District'])
    
    def get_market_locations(self, state: str) -> List[Dict]:
        """Get market locations in state"""
        districts = self.get_nearby_districts(state)
        return [
            {'name': f'{district} Market', 'district': district, 'state': state}
            for district in districts[:5]
        ]