"""
src/search_engine.py
====================
API-Compatible Search and Filter Engine
"""

import json
from typing import List, Dict, Optional


class CaseSearchEngine:
    """
    Search and filter legal judgments
    Compatible with Flask API
    """
    
    def __init__(self, judgments_file: str = "output/judgments.json"):
        self.judgments_file = judgments_file
        self.cases = []
        self.load_cases()
    
    def load_cases(self):
        """Load cases from JSON file"""
        try:
            with open(self.judgments_file, 'r', encoding='utf-8') as f:
                self.cases = json.load(f)
            print(f"✅ Loaded {len(self.cases)} cases for search")
        except FileNotFoundError:
            print(f"⚠️ No judgments file found at {self.judgments_file}")
            self.cases = []
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {self.judgments_file}")
            self.cases = []
    
    def search(self,
               keyword: Optional[str] = None,
               court: Optional[str] = None,
               outcome: Optional[str] = None,
               case_type: Optional[str] = None,
               act: Optional[str] = None) -> List[Dict]:
        """
        Search cases by multiple criteria (API compatible)
        
        Parameters:
        - keyword: Search in case_id, summary, facts, legal_issues, etc.
        - court: Filter by court name
        - outcome: Filter by predicted_outcome
        - case_type: Filter by case_type
        - act: Filter by acts_referred
        
        Returns:
        - List of matching cases
        """
        results = self.cases.copy()
        
        # Filter by keyword (searches multiple fields)
        if keyword:
            keyword_lower = keyword.lower()
            results = [
                case for case in results
                if self._matches_keyword(case, keyword_lower)
            ]
        
        # Filter by court
        if court:
            court_lower = court.lower()
            results = [
                case for case in results
                if court_lower in case.get('court', '').lower()
            ]
        
        # Filter by outcome
        if outcome:
            outcome_lower = outcome.lower()
            results = [
                case for case in results
                if case.get('predicted_outcome', '').lower() == outcome_lower
            ]
        
        # Filter by case type
        if case_type:
            case_type_lower = case_type.lower()
            results = [
                case for case in results
                if case_type_lower in case.get('case_type', '').lower()
            ]
        
        # Filter by act
        if act:
            act_lower = act.lower()
            results = [
                case for case in results
                if self._matches_act(case, act_lower)
            ]
        
        return results
    
    def _matches_keyword(self, case: Dict, keyword: str) -> bool:
        """
        Check if case matches keyword in any relevant field
        Searches: case_id, summary, facts, primary_legal_issue, legal_issues,
                 judges, petitioners, respondents, acts_referred
        """
        # Search in string fields
        searchable_fields = [
            'case_id',
            'summary',
            'facts',
            'primary_legal_issue',
            'judgment_reasoning',
            'arguments',
            'court'
        ]
        
        for field in searchable_fields:
            value = case.get(field, '')
            if isinstance(value, str) and keyword in value.lower():
                return True
        
        # Search in list fields
        list_fields = [
            'judges',
            'petitioners',
            'respondents',
            'acts_referred',
            'legal_issues',
            'key_legal_points'
        ]
        
        for field in list_fields:
            values = case.get(field, [])
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, str) and keyword in item.lower():
                        return True
            elif isinstance(values, str) and keyword in values.lower():
                return True
        
        return False
    
    def _matches_act(self, case: Dict, act: str) -> bool:
        """Check if case references the specified act"""
        acts = case.get('acts_referred', [])
        
        # Handle both list and string
        if isinstance(acts, str):
            return act in acts.lower()
        elif isinstance(acts, list):
            return any(act in a.lower() for a in acts if isinstance(a, str))
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded cases"""
        stats = {
            'total': len(self.cases),
            'by_court': {},
            'by_outcome': {},
            'by_case_type': {}
        }
        
        for case in self.cases:
            # Count by court
            court = case.get('court', 'Unknown')
            stats['by_court'][court] = stats['by_court'].get(court, 0) + 1
            
            # Count by outcome
            outcome = case.get('predicted_outcome', 'Unknown')
            stats['by_outcome'][outcome] = stats['by_outcome'].get(outcome, 0) + 1
            
            # Count by case type
            case_type = case.get('case_type', 'Unknown')
            stats['by_case_type'][case_type] = stats['by_case_type'].get(case_type, 0) + 1
        
        return stats
    
    def get_case_by_id(self, case_id: str) -> Optional[Dict]:
        """Get a specific case by ID"""
        for case in self.cases:
            if case.get('case_id') == case_id:
                return case
        return None


if __name__ == "__main__":
    # Test the search engine
    print("="*70)
    print("Testing Case Search Engine")
    print("="*70)
    
    engine = CaseSearchEngine()
    print(f"\nLoaded {len(engine.cases)} cases")
    
    # Test statistics
    stats = engine.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total cases: {stats['total']}")
    print(f"  Courts: {stats['by_court']}")
    print(f"  Outcomes: {stats['by_outcome']}")
    
    # Test search
    print(f"\n--- Testing Keyword Search ---")
    results = engine.search(keyword="IPC 302")
    print(f"Found {len(results)} cases matching 'IPC 302'")
    
    print(f"\n--- Testing Court Filter ---")
    results = engine.search(court="Delhi")
    print(f"Found {len(results)} cases from Delhi High Court")
    
    print(f"\n--- Testing Outcome Filter ---")
    results = engine.search(outcome="Allowed")
    print(f"Found {len(results)} cases with outcome 'Allowed'")
    
    print(f"\n--- Testing Combined Filters ---")
    results = engine.search(court="Delhi", outcome="Dismissed")
    print(f"Found {len(results)} dismissed cases from Delhi")
