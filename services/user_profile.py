import streamlit as st
import json
from typing import Dict, Any, List, Optional

class UserProfileManager:
    def __init__(self):
        self.personal_keywords = {
            'age': ['age', 'old', 'young', 'retirement', 'years old'],
            'income': ['income', 'salary', 'earn', 'make money', 'budget'],
            'risk_tolerance': ['risk', 'conservative', 'aggressive', 'moderate'],
            'investment_goals': ['goals', 'target', 'objective', 'plan'],
            'time_horizon': ['when', 'timeline', 'years', 'months', 'long term', 'short term'],
            'location': ['country', 'location', 'where', 'tax', 'jurisdiction']
        }
    
    def collect_user_info(self, question: str) -> List[str]:
        """Identify what user information is needed based on the question"""
        needed_info = []
        for info_type, keywords in self.personal_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                needed_info.append(info_type)
        return needed_info
    
    def get_user_input(self, info_type: str) -> Optional[str]:
        """Get user input for specific information type"""
        prompts = {
            'age': "What's your age range? (20-30, 30-40, 40-50, 50-60, 60+)",
            'income': "What's your approximate annual income range? (Optional)",
            'risk_tolerance': "What's your risk tolerance? (Conservative, Moderate, Aggressive)",
            'investment_goals': "What are your main investment goals?",
            'time_horizon': "What's your investment timeline?",
            'location': "What country/region are you in? (for tax considerations)"
        }
        
        if info_type in prompts:
            return st.text_input(prompts[info_type], key=f"user_{info_type}")
        return None
    
    def collect_and_process_user_info(self, question: str) -> str:
        """Collect user information and return context string"""
        needed_info = self.collect_user_info(question)
        
        user_context = ""
        if needed_info:
            st.markdown('<h3 class="section-header">👤 I need some information to provide personalized advice:</h3>', unsafe_allow_html=True)
            
            collected_info = {}
            for info_type in needed_info:
                if info_type not in st.session_state.user_profile:
                    user_input = self.get_user_input(info_type)
                    if user_input:
                        collected_info[info_type] = user_input
                        st.session_state.user_profile[info_type] = user_input
            
            if st.session_state.user_profile:
                user_context = f"\nUser Profile: {json.dumps(st.session_state.user_profile, indent=2)}\n"
        
        return user_context
    
    def calculate_risk_score(self) -> int:
        """Calculate user's risk score based on profile"""
        if not st.session_state.user_profile:
            return 5
        
        risk_tolerance = st.session_state.user_profile.get('risk_tolerance', 'Moderate')
        age_range = st.session_state.user_profile.get('age', 'Unknown')
        time_horizon = st.session_state.user_profile.get('time_horizon', 'Unknown')
        
        risk_score = 5
        
        if risk_tolerance.lower() == 'conservative':
            risk_score = 3
        elif risk_tolerance.lower() == 'aggressive':
            risk_score = 8
        
        if '20-30' in age_range or '30-40' in age_range:
            risk_score += 1
        elif '50-60' in age_range or '60+' in age_range:
            risk_score -= 1
        
        if 'long term' in time_horizon.lower() or 'years' in time_horizon.lower():
            risk_score += 1
        elif 'short term' in time_horizon.lower() or 'months' in time_horizon.lower():
            risk_score -= 1
        
        return max(1, min(10, risk_score))