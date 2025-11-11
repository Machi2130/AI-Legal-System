"""
src/preprocess.py
=================
Clean and normalize extracted legal data
"""

import re
from typing import List, Dict
from datetime import datetime

class DataPreprocessor:
    """
    Preprocess and clean legal data
    """
    
    def __init__(self):
        print("✅ DataPreprocessor initialized")
    
    def validate_case(self, case: Dict) -> bool:
        """Validate case has minimum required fields"""
        
        required = ['case_id', 'court', 'summary', 'full_judgment_text']
        
        for field in required:
            if field not in case or not case[field]:
                return False
        
        # Validate date format
        date = case.get('date', '')
        if date and date != 'Unknown':
            if not re.match(r'\d{4}-\d{2}-\d{2}', date):
                return False
        
        # Validate outcome
        outcome = case.get('predicted_outcome', '')
        valid_outcomes = ['Allowed', 'Dismissed', 'Partly Allowed', 'Disposed', 'Unknown']
        if outcome not in valid_outcomes:
            return False
        
        return True
    
    def clean_case(self, case: Dict) -> Dict:
        """Clean and normalize case data"""
        
        # Normalize arrays
        array_fields = [
            'petitioners', 'respondents', 'judges', 'acts_referred',
            'sections', 'key_legal_points', 'precedents_cited',
            'legal_issues', 'citations'
        ]
        
        for field in array_fields:
            if field not in case:
                case[field] = []
            elif not isinstance(case[field], list):
                case[field] = []
        
        # Normalize strings
        string_fields = [
            'case_id', 'court', 'case_type', 'facts',
            'arguments', 'judgment_reasoning'
        ]
        
        for field in string_fields:
            if field not in case or not case[field]:
                case[field] = "Unknown"
        
        # Clean text
        if 'summary' in case:
            case['summary'] = case['summary'].strip()
        
        if 'full_judgment_text' in case:
            case['full_judgment_text'] = case['full_judgment_text'].strip()
        
        return case
    
    def remove_duplicates(self, cases: List[Dict]) -> List[Dict]:
        """Remove duplicate cases"""
        
        seen = set()
        unique = []
        
        for case in cases:
            case_id = case.get('case_id')
            if case_id and case_id not in seen:
                seen.add(case_id)
                unique.append(case)
        
        return unique
    
    def process(self, cases: List[Dict]) -> List[Dict]:
        """Process all cases"""
        
        print(f"\n{'='*70}")
        print(f"🔧 PREPROCESSING DATA")
        print(f"{'='*70}")
        print(f"Input cases: {len(cases)}")
        
        # Validate
        valid_cases = []
        for case in cases:
            if self.validate_case(case):
                cleaned = self.clean_case(case)
                valid_cases.append(cleaned)
        
        print(f"Valid cases: {len(valid_cases)}")
        
        # Remove duplicates
        unique_cases = self.remove_duplicates(valid_cases)
        print(f"Unique cases: {len(unique_cases)}")
        
        print(f"✅ Preprocessing complete")
        return unique_cases


if __name__ == "__main__":
    # Test
    preprocessor = DataPreprocessor()
    print("Preprocessor ready")
