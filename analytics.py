"""
src/analytics.py
================
Legal Analytics Engine - API-Compatible Version
Provides statistical insights for the dashboard
"""

from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List
import json
import os


class LegalAnalytics:
    """
    Analytics engine for legal case statistics
    Compatible with Flask API endpoints
    """
    
    def __init__(self, judgments_file: str = "output/judgments.json"):
        self.judgments_file = judgments_file
        self.cases = []
        self._load_cases()
    
    def _load_cases(self):
        """Load cases from JSON file"""
        try:
            if os.path.exists(self.judgments_file):
                with open(self.judgments_file, 'r', encoding='utf-8') as f:
                    self.cases = json.load(f)
            else:
                self.cases = []
        except (FileNotFoundError, json.JSONDecodeError):
            self.cases = []
    
    # ============================================
    # API-COMPATIBLE METHODS
    # ============================================
    
    def judge_statistics(self, top_n: int = 20) -> List[Dict]:
        """
        Get judge statistics (API endpoint compatible)
        Returns: [{"judge": "Name", "total_cases": 10, "outcomes": {...}}, ...]
        """
        judge_stats = defaultdict(lambda: {
            'total_cases': 0,
            'outcomes': defaultdict(int)
        })
        
        for case in self.cases:
            judges = case.get('judges', [])
            outcome = case.get('predicted_outcome', 'Unknown')
            
            # Handle both list and string judges
            if isinstance(judges, str):
                judges = [judges]
            
            for judge in judges:
                if judge and judge.strip():
                    judge_stats[judge]['total_cases'] += 1
                    judge_stats[judge]['outcomes'][outcome] += 1
        
        # Convert to list format
        result = []
        for judge, stats in judge_stats.items():
            result.append({
                'judge': judge,
                'total_cases': stats['total_cases'],
                'outcomes': dict(stats['outcomes'])
            })
        
        # Sort by total cases
        result.sort(key=lambda x: x['total_cases'], reverse=True)
        
        return result[:top_n]
    
    def most_cited_acts(self, top_n: int = 20) -> List[Dict]:
        """
        Get most cited acts (API endpoint compatible)
        Returns: [{"act": "IPC", "count": 50}, ...]
        """
        act_counter = Counter()
        
        for case in self.cases:
            acts = case.get('acts_referred', [])
            
            # Handle both list and string
            if isinstance(acts, str):
                acts = [acts]
            
            for act in acts:
                if act and act.strip():
                    act_counter[act] += 1
        
        # Convert to list format
        result = [
            {'act': act, 'count': count}
            for act, count in act_counter.most_common(top_n)
        ]
        
        return result
    
    def court_distribution(self) -> List[Dict]:
        """
        Get court distribution (API endpoint compatible)
        Returns: [{"court": "Delhi High Court", "count": 100}, ...]
        """
        court_counter = Counter()
        
        for case in self.cases:
            court = case.get('court', 'Unknown')
            if court:
                court_counter[court] += 1
        
        # Convert to list format
        result = [
            {'court': court, 'count': count}
            for court, count in court_counter.items()
        ]
        
        # Sort by count
        result.sort(key=lambda x: x['count'], reverse=True)
        
        return result
    
    def outcome_distribution(self) -> List[Dict]:
        """
        Get outcome distribution (API endpoint compatible)
        Returns: [{"outcome": "Allowed", "count": 50}, ...]
        """
        outcome_counter = Counter()
        
        for case in self.cases:
            outcome = case.get('predicted_outcome', 'Unknown')
            if outcome:
                outcome_counter[outcome] += 1
        
        # Convert to list format
        result = [
            {'outcome': outcome, 'count': count}
            for outcome, count in outcome_counter.items()
        ]
        
        # Sort by count
        result.sort(key=lambda x: x['count'], reverse=True)
        
        return result
    
    # ============================================
    # ADVANCED ANALYTICS (Optional)
    # ============================================
    
    def case_type_distribution(self) -> List[Dict]:
        """Get distribution of case types"""
        type_counter = Counter()
        
        for case in self.cases:
            case_type = case.get('case_type', 'Unknown')
            if case_type:
                type_counter[case_type] += 1
        
        return [
            {'case_type': ctype, 'count': count}
            for ctype, count in type_counter.most_common()
        ]
    
    def temporal_analysis(self) -> Dict[str, int]:
        """Get cases per year"""
        year_counter = Counter()
        
        for case in self.cases:
            date = case.get('date', 'Unknown')
            if date and date != 'Unknown' and len(date) >= 4:
                year = date[:4]
                year_counter[year] += 1
        
        return dict(sorted(year_counter.items()))
    
    def judge_outcome_rates(self, judge_name: str) -> Dict:
        """Get outcome statistics for a specific judge"""
        total_cases = 0
        outcomes = Counter()
        
        for case in self.cases:
            judges = case.get('judges', [])
            if isinstance(judges, str):
                judges = [judges]
            
            if judge_name in judges:
                total_cases += 1
                outcome = case.get('predicted_outcome', 'Unknown')
                outcomes[outcome] += 1
        
        if total_cases == 0:
            return {}
        
        return {
            'judge': judge_name,
            'total_cases': total_cases,
            'outcomes': dict(outcomes),
            'allowed_rate': (outcomes.get('Allowed', 0) / total_cases) * 100,
            'dismissed_rate': (outcomes.get('Dismissed', 0) / total_cases) * 100
        }
    
    def generate_summary(self) -> Dict:
        """Generate complete analytics summary"""
        return {
            'metadata': {
                'total_cases': len(self.cases),
                'generated_at': datetime.now().isoformat()
            },
            'top_judges': self.judge_statistics(10),
            'top_acts': self.most_cited_acts(10),
            'courts': self.court_distribution(),
            'outcomes': self.outcome_distribution(),
            'case_types': self.case_type_distribution(),
            'temporal': self.temporal_analysis()
        }


if __name__ == "__main__":
    # Test the analytics
    analytics = LegalAnalytics()
    
    print("="*70)
    print("LEGAL ANALYTICS SUMMARY")
    print("="*70)
    
    print(f"\nTotal Cases: {len(analytics.cases)}")
    
    print("\nTop 5 Judges:")
    for judge in analytics.judge_statistics(5):
        print(f"  - {judge['judge']}: {judge['total_cases']} cases")
    
    print("\nTop 5 Acts:")
    for act in analytics.most_cited_acts(5):
        print(f"  - {act['act']}: {act['count']} citations")
    
    print("\nCourt Distribution:")
    for court in analytics.court_distribution():
        print(f"  - {court['court']}: {court['count']} cases")
    
    print("\nOutcome Distribution:")
    for outcome in analytics.outcome_distribution():
        print(f"  - {outcome['outcome']}: {outcome['count']} cases")
