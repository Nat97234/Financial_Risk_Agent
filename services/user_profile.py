import streamlit as st
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re
import logging
from config.settings import *

logger = logging.getLogger(__name__)

class EnhancedUserProfileManager:
    def __init__(self):
        self.personal_keywords = {
            'age': {
                'en': ['age', 'old', 'young', 'retirement', 'years old', 'born', 'birth'],
                'ar': ['عمر', 'سن', 'عام', 'تقاعد', 'ولادة', 'مولود']
            },
            'income': {
                'en': ['income', 'salary', 'earn', 'make money', 'budget', 'wages', 'revenue'],
                'ar': ['دخل', 'راتب', 'كسب', 'مرتب', 'أجر', 'إيرادات', 'ميزانية']
            },
            'risk_tolerance': {
                'en': ['risk', 'conservative', 'aggressive', 'moderate', 'volatility', 'safety'],
                'ar': ['مخاطرة', 'محافظ', 'عدواني', 'معتدل', 'أمان', 'استقرار']
            },
            'investment_goals': {
                'en': ['goals', 'target', 'objective', 'plan', 'purpose', 'aim'],
                'ar': ['هدف', 'غاية', 'خطة', 'قصد', 'غرض', 'مرمى']
            },
            'time_horizon': {
                'en': ['when', 'timeline', 'years', 'months', 'long term', 'short term', 'horizon'],
                'ar': ['متى', 'وقت', 'سنة', 'شهر', 'طويل الأمد', 'قصير الأمد', 'مدى']
            },
            'location': {
                'en': ['country', 'location', 'where', 'tax', 'jurisdiction', 'region'],
                'ar': ['دولة', 'مكان', 'أين', 'ضريبة', 'منطقة', 'إقليم']
            },
            'experience': {
                'en': ['experience', 'beginner', 'expert', 'novice', 'professional', 'seasoned'],
                'ar': ['خبرة', 'مبتدئ', 'خبير', 'محترف', 'متمرس', 'ماهر']
            },
            'investment_amount': {
                'en': ['invest', 'amount', 'capital', 'money', 'budget', 'fund'],
                'ar': ['استثمار', 'مبلغ', 'رأس مال', 'مال', 'ميزانية', 'صندوق']
            }
        }

    def detect_language(self, text: str) -> str:
        """Detect if text is in Arabic or English"""
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        if len(arabic_chars) > len(text) * 0.3:  # If 30% or more are Arabic characters
            return 'ar'
        return 'en'

    def get_localized_text(self, key: str, lang: str = 'en') -> str:
        """Get localized text based on language"""
        texts = {
            'user_profile_section': {
                'en': '👤 User Profile Information Detected',
                'ar': '👤 تم اكتشاف معلومات المستخدم'
            },
            'risk_assessment': {
                'en': '📊 Risk Profile Analysis',
                'ar': '📊 تحليل الملف الشخصي للمخاطر'
            }
        }
        return texts.get(key, {}).get(lang, texts.get(key, {}).get('en', key))

    def collect_user_info(self, question: str) -> Tuple[List[str], str]:
        """Identify what user information is needed and detect language - NO USER INPUT"""
        lang = self.detect_language(question)
        needed_info = []
        
        question_lower = question.lower()
        
        for info_type, keywords_dict in self.personal_keywords.items():
            keywords = keywords_dict.get(lang, keywords_dict.get('en', []))
            if any(keyword in question_lower for keyword in keywords):
                needed_info.append(info_type)
        
        return needed_info, lang

    def extract_profile_info_from_text(self, text: str, lang: str = 'en') -> Dict[str, Any]:
        """Extract profile information from user's text without asking questions"""
        extracted_info = {}
        text_lower = text.lower()
        
        try:
            # Extract age information
            age_patterns = {
                'en': [r'(\d+)\s*years?\s*old', r'age\s*(?:is\s*)?(\d+)', r'(\d+)\s*year'],
                'ar': [r'عمري\s*(\d+)', r'عام\s*(\d+)', r'سن\s*(\d+)']
            }
            
            for pattern in age_patterns.get(lang, age_patterns['en']):
                match = re.search(pattern, text_lower)
                if match:
                    age = int(match.group(1))
                    if 18 <= age <= 100:
                        if age < 26:
                            extracted_info['age'] = '18-25'
                        elif age < 36:
                            extracted_info['age'] = '26-35'
                        elif age < 46:
                            extracted_info['age'] = '36-45'
                        elif age < 56:
                            extracted_info['age'] = '46-55'
                        elif age < 66:
                            extracted_info['age'] = '56-65'
                        else:
                            extracted_info['age'] = '65+'
                    break
            
            # Extract risk tolerance
            risk_indicators = {
                'conservative': ['conservative', 'safe', 'low risk', 'stable', 'secure', 'محافظ', 'آمن'],
                'moderate': ['moderate', 'balanced', 'medium risk', 'معتدل', 'متوازن'],
                'aggressive': ['aggressive', 'high risk', 'risky', 'growth', 'عدواني', 'مخاطر عالية']
            }
            
            for risk_level, indicators in risk_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    extracted_info['risk_tolerance'] = risk_level.title()
                    break
            
            # Extract time horizon
            if any(term in text_lower for term in ['long term', 'long-term', 'years', 'decade', 'طويل الأمد']):
                extracted_info['time_horizon'] = '7+ years'
            elif any(term in text_lower for term in ['short term', 'short-term', 'months', 'قصير الأمد']):
                extracted_info['time_horizon'] = '1-3 years'
            elif any(term in text_lower for term in ['medium term', 'few years']):
                extracted_info['time_horizon'] = '3-7 years'
            
            # Extract investment goals
            goal_keywords = {
                'retirement': ['retirement', 'retire', 'pension', 'تقاعد'],
                'wealth building': ['wealth', 'build', 'grow', 'accumulate', 'بناء الثروة'],
                'income': ['income', 'dividend', 'cash flow', 'دخل'],
                'education': ['education', 'college', 'university', 'school', 'تعليم'],
                'home': ['house', 'home', 'property', 'منزل', 'عقار']
            }
            
            detected_goals = []
            for goal, keywords in goal_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected_goals.append(goal.title())
            
            if detected_goals:
                extracted_info['investment_goals'] = detected_goals
            
            # Extract experience level
            experience_keywords = {
                'beginner': ['beginner', 'new', 'start', 'first time', 'مبتدئ', 'جديد'],
                'experienced': ['experienced', 'years of', 'trading', 'investing', 'خبرة', 'متمرس'],
                'expert': ['expert', 'professional', 'advanced', 'خبير', 'محترف']
            }
            
            for level, keywords in experience_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    extracted_info['experience'] = level.title()
                    break
            
            # Extract investment amount mentions
            amount_patterns = [
                r'\$([0-9,]+)',
                r'([0-9,]+)\s*(?:dollar|usd|$)',
                r'([0-9,]+)k',
                r'([0-9,]+)\s*thousand'
            ]
            
            for pattern in amount_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        amount = float(amount_str)
                        if 'k' in pattern or 'thousand' in pattern:
                            amount *= 1000
                        
                        if amount < 25000:
                            extracted_info['investment_amount'] = '<$25K'
                        elif amount < 50000:
                            extracted_info['investment_amount'] = '$25K-$50K'
                        elif amount < 100000:
                            extracted_info['investment_amount'] = '$50K-$100K'
                        elif amount < 250000:
                            extracted_info['investment_amount'] = '$100K-$250K'
                        else:
                            extracted_info['investment_amount'] = '$250K+'
                    except ValueError:
                        pass
                    break
            
        except Exception as e:
            logger.error(f"Error extracting profile info: {e}")
        
        return extracted_info

    def calculate_comprehensive_risk_score(self, profile_info: Dict[str, Any] = None) -> float:
        """Calculate comprehensive risk score from available profile data"""
        if profile_info is None:
            profile_info = st.session_state.get('user_profile', {})
        
        if not profile_info:
            return 5.0  # Default moderate risk score
        
        score_components = []
        
        # Age-based risk capacity
        age_range = profile_info.get('age', '')
        if '18-25' in age_range or '26-35' in age_range:
            score_components.append(8)  # Young = higher risk capacity
        elif '36-45' in age_range:
            score_components.append(7)
        elif '46-55' in age_range:
            score_components.append(5)
        elif '56-65' in age_range:
            score_components.append(3)
        else:  # 65+
            score_components.append(2)
        
        # Income-based risk capacity
        income = profile_info.get('income', '')
        if '$250K+' in income or '250' in income:
            score_components.append(9)
        elif '$100K-$250K' in income or '100-250' in income:
            score_components.append(7)
        elif '$50K-$100K' in income or '50-100' in income:
            score_components.append(5)
        else:
            score_components.append(3)
        
        # Time horizon
        horizon = profile_info.get('time_horizon', '')
        if '7+' in horizon or 'long term' in horizon.lower():
            score_components.append(8)
        elif '3-7' in horizon:
            score_components.append(6)
        elif '1-3' in horizon:
            score_components.append(4)
        else:
            score_components.append(2)
        
        # Experience level
        experience = profile_info.get('experience', '')
        if 'Expert' in experience or 'خبير' in experience:
            score_components.append(8)
        elif 'Experienced' in experience or 'خبرة' in experience:
            score_components.append(6)
        elif 'Some' in experience or 'بعض' in experience:
            score_components.append(4)
        else:
            score_components.append(2)
        
        # Risk tolerance stated preference
        risk_tolerance = profile_info.get('risk_tolerance', '')
        if 'aggressive' in risk_tolerance.lower() or 'عدواني' in risk_tolerance:
            score_components.append(9)
        elif 'moderate' in risk_tolerance.lower() or 'معتدل' in risk_tolerance:
            score_components.append(5)
        elif 'conservative' in risk_tolerance.lower() or 'محافظ' in risk_tolerance:
            score_components.append(2)
        
        # Calculate weighted average
        if score_components:
            return max(1, min(10, sum(score_components) / len(score_components)))
        else:
            return 5.0

    def categorize_risk_level(self, risk_score: float) -> Dict[str, Any]:
        """Categorize risk level with detailed explanation"""
        if risk_score <= 3:
            category = "CONSERVATIVE"
        elif risk_score <= 6:
            category = "MODERATE"
        elif risk_score <= 8:
            category = "AGGRESSIVE"
        else:
            category = "SPECULATIVE"
        
        category_info = RISK_CATEGORIES[category].copy()
        category_info['score'] = risk_score
        category_info['name'] = category
        
        return category_info

    def generate_personalized_context(self, question: str) -> str:
        """Generate personalized context for AI responses without asking user input"""
        # Extract information from the question itself
        lang = self.detect_language(question)
        extracted_info = self.extract_profile_info_from_text(question, lang)
        
        # Update session state with extracted info
        if extracted_info:
            if 'user_profile' not in st.session_state:
                st.session_state.user_profile = {}
            st.session_state.user_profile.update(extracted_info)
            st.session_state.user_profile['language'] = lang
            st.session_state.user_profile['last_updated'] = datetime.now().isoformat()
        
        # Get current profile
        profile = st.session_state.get('user_profile', {})
        
        if not profile:
            # Return basic context without profile
            if lang == 'ar':
                return "يرجى الإجابة باللغة العربية. قدم نصائح مالية عامة مناسبة للمستثمرين المعتدلين."
            else:
                return "Please provide general financial advice suitable for moderate investors."
        
        context_parts = []
        
        # Language preference
        if lang == 'ar':
            context_parts.append("Please respond in Arabic language (العربية).")
        else:
            context_parts.append("Please respond in English.")
        
        # User profile summary
        context_parts.append("User Profile Summary:")
        
        # Demographics
        age = profile.get('age', 'Not specified')
        income = profile.get('income', 'Not specified')
        location = profile.get('location', 'Not specified')
        context_parts.append(f"- Age: {age}, Income: {income}, Location: {location}")
        
        # Investment profile
        experience = profile.get('experience', 'Not specified')
        horizon = profile.get('time_horizon', 'Not specified')
        context_parts.append(f"- Investment Experience: {experience}, Time Horizon: {horizon}")
        
        # Risk profile
        risk_score = self.calculate_comprehensive_risk_score(profile)
        risk_category = self.categorize_risk_level(risk_score)
        context_parts.append(f"- Risk Score: {risk_score:.1f}/10 ({risk_category['name']})")
        
        # Investment goals
        goals = profile.get('investment_goals', [])
        if goals:
            if isinstance(goals, list):
                goals_str = ", ".join(goals)
            else:
                goals_str = str(goals)
            context_parts.append(f"- Investment Goals: {goals_str}")
        
        # Investment amount if specified
        investment_amount = profile.get('investment_amount', '')
        if investment_amount:
            context_parts.append(f"- Available Investment Amount: {investment_amount}")
        
        context_parts.append("\nPlease provide advice that is:")
        context_parts.append("- Appropriate for their risk tolerance and capacity")
        context_parts.append("- Aligned with their investment timeline and goals")
        context_parts.append("- Suitable for their experience level")
        context_parts.append("- Culturally and linguistically appropriate")
        
        return "\n".join(context_parts)

    def collect_and_process_user_info(self, question: str) -> str:
        """Main method to process user information without asking questions"""
        needed_info, lang = self.collect_user_info(question)
        
        # Simply generate personalized context based on available information
        return self.generate_personalized_context(question)

    def get_investment_recommendations_framework(self) -> Dict[str, Any]:
        """Get framework for investment recommendations based on user profile"""
        profile = st.session_state.get('user_profile', {})
        
        if not profile:
            # Default moderate allocation
            return {
                'asset_allocation': {'stocks': 50, 'bonds': 35, 'cash': 10, 'alternatives': 5},
                'geographic_allocation': {'domestic': 60, 'developed_international': 25, 'emerging_markets': 15},
                'sector_focus': ['Diversified Index Funds'],
                'rebalancing_frequency': 'Semi-annually',
                'risk_monitoring': True
            }
        
        risk_score = self.calculate_comprehensive_risk_score(profile)
        
        # Asset allocation recommendations
        if risk_score <= 3:  # Conservative
            allocation = {'stocks': 20, 'bonds': 60, 'cash': 15, 'alternatives': 5}
        elif risk_score <= 6:  # Moderate
            allocation = {'stocks': 50, 'bonds': 35, 'cash': 10, 'alternatives': 5}
        elif risk_score <= 8:  # Aggressive
            allocation = {'stocks': 75, 'bonds': 15, 'cash': 5, 'alternatives': 5}
        else:  # Speculative
            allocation = {'stocks': 85, 'bonds': 5, 'cash': 5, 'alternatives': 5}
        
        # Geographic diversification
        geo_allocation = {'domestic': 60, 'developed_international': 25, 'emerging_markets': 15}
        
        # Sector preferences based on goals
        goals = profile.get('investment_goals', [])
        sector_focus = []
        
        if 'Income' in str(goals) or 'توليد الدخل' in str(goals):
            sector_focus.extend(['Utilities', 'REITs', 'Dividend Stocks'])
        if 'Wealth' in str(goals) or 'بناء الثروة' in str(goals):
            sector_focus.extend(['Technology', 'Growth Stocks'])
        if 'Conservative' in str(goals) or 'حفظ رأس المال' in str(goals):
            sector_focus.extend(['Bonds', 'Blue Chip Stocks'])
        
        if not sector_focus:
            sector_focus = ['Diversified Index Funds']
        
        return {
            'asset_allocation': allocation,
            'geographic_allocation': geo_allocation,
            'sector_focus': sector_focus,
            'rebalancing_frequency': 'Quarterly' if risk_score > 6 else 'Semi-annually',
            'risk_monitoring': True
        }

    def display_profile_summary(self):
        """Display user profile summary if available"""
        profile = st.session_state.get('user_profile', {})
        
        if not profile:
            st.info("No user profile information available yet. Information will be extracted from your questions automatically.")
            return
        
        lang = profile.get('language', 'en')
        
        st.markdown("### 👤 Profile Summary" if lang == 'en' else "### 👤 ملخص الملف الشخصي")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Age:** {profile.get('age', 'Not detected')}")
            st.write(f"**Experience:** {profile.get('experience', 'Not detected')}")
            st.write(f"**Time Horizon:** {profile.get('time_horizon', 'Not detected')}")
            
        with col2:
            risk_score = self.calculate_comprehensive_risk_score(profile)
            risk_category = self.categorize_risk_level(risk_score)
            st.write(f"**Risk Score:** {risk_score:.1f}/10")
            st.write(f"**Risk Category:** {risk_category['name']}")
            st.write(f"**Language:** {'Arabic' if lang == 'ar' else 'English'}")
        
        # Goals if available
        goals = profile.get('investment_goals', [])
        if goals:
            if isinstance(goals, list):
                goals_str = ", ".join(goals)
            else:
                goals_str = str(goals)
            st.write(f"**Goals:** {goals_str}")
        
        st.info("ℹ️ Profile information is automatically extracted from your questions. No manual input required!" 
               if lang == 'en' else "ℹ️ يتم استخراج معلومات الملف الشخصي تلقائياً من أسئلتك. لا حاجة لإدخال يدوي!")

    def clear_profile(self):
        """Clear user profile"""
        st.session_state.user_profile = {}
        lang = st.session_state.get('app_settings', {}).get('language', 'en')
        st.success("User profile cleared!" if lang == 'en' else "تم مسح الملف الشخصي!")

    def export_user_profile(self) -> Dict[str, Any]:
        """Export user profile for analysis or backup"""
        profile = st.session_state.get('user_profile', {})
        
        if not profile:
            return {}
        
        # Add computed metrics
        export_profile = profile.copy()
        export_profile['export_timestamp'] = datetime.now().isoformat()
        export_profile['risk_score'] = self.calculate_comprehensive_risk_score(profile)
        export_profile['risk_category'] = self.categorize_risk_level(export_profile['risk_score'])
        export_profile['investment_framework'] = self.get_investment_recommendations_framework()
        
        return export_profile