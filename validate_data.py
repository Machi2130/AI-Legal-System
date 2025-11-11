"""
validate_data.py
================
Purpose: Validate that all required fields are properly stored in judgments.json
Usage: python validate_data.py
"""

import json
from typing import Dict, List
from collections import Counter


def validate_case(case: Dict, case_idx: int) -> List[str]:
    """Validate a single case and return list of issues"""
    issues = []
    
    # Required fields
    required_fields = [
        "case_id", "court", "date", "petitioners", "respondents",
        "judges", "acts_referred", "predicted_outcome", "summary", "source"
    ]
    
    for field in required_fields:
        if field not in case:
            issues.append(f"Case {case_idx}: Missing field '{field}'")
        elif not case[field]:
            issues.append(f"Case {case_idx}: Empty field '{field}'")
    
    # Type validation
    list_fields = ["petitioners", "respondents", "judges", "acts_referred"]
    for field in list_fields:
        if field in case and not isinstance(case[field], list):
            issues.append(f"Case {case_idx}: '{field}' should be list, got {type(case[field])}")
    
    # Summary length
    if "summary" in case and len(case.get("summary", "")) < 50:
        issues.append(f"Case {case_idx}: Summary too short ({len(case.get('summary', ''))} chars)")
    
    # Acts validation
    if "acts_referred" in case:
        if len(case["acts_referred"]) == 0:
            issues.append(f"Case {case_idx}: No acts extracted")
        elif len(case["acts_referred"]) > 20:
            issues.append(f"Case {case_idx}: Too many acts ({len(case['acts_referred'])})")
    
    return issues


def analyze_dataset(cases: List[Dict]) -> Dict:
    """Analyze the complete dataset"""
    
    stats = {
        "total_cases": len(cases),
        "valid_cases": 0,
        "invalid_cases": 0,
        "fields_stats": {},
        "acts_stats": {
            "cases_with_acts": 0,
            "cases_without_acts": 0,
            "total_acts": 0,
            "unique_acts": set(),
            "most_common_acts": []
        },
        "outcome_distribution": Counter(),
        "court_distribution": Counter(),
        "year_distribution": Counter(),
        "summary_lengths": []
    }
    
    all_issues = []
    
    for idx, case in enumerate(cases):
        # Validate case
        issues = validate_case(case, idx)
        if issues:
            all_issues.extend(issues)
            stats["invalid_cases"] += 1
        else:
            stats["valid_cases"] += 1
        
        # Acts statistics
        acts = case.get("acts_referred", [])
        if acts:
            stats["acts_stats"]["cases_with_acts"] += 1
            stats["acts_stats"]["total_acts"] += len(acts)
            stats["acts_stats"]["unique_acts"].update(acts)
        else:
            stats["acts_stats"]["cases_without_acts"] += 1
        
        # Outcome distribution
        outcome = case.get("predicted_outcome", "Unknown")
        stats["outcome_distribution"][outcome] += 1
        
        # Court distribution
        court = case.get("court", "Unknown")
        stats["court_distribution"][court] += 1
        
        # Year distribution
        date = case.get("date", "Unknown")
        if date != "Unknown" and len(date) >= 4:
            year = date[:4]
            stats["year_distribution"][year] += 1
        
        # Summary length
        summary = case.get("summary", "")
        stats["summary_lengths"].append(len(summary))
    
    # Calculate act statistics
    stats["acts_stats"]["unique_acts_count"] = len(stats["acts_stats"]["unique_acts"])
    stats["acts_stats"]["avg_acts_per_case"] = (
        stats["acts_stats"]["total_acts"] / max(stats["total_cases"], 1)
    )
    
    # Most common acts
    act_counter = Counter()
    for case in cases:
        act_counter.update(case.get("acts_referred", []))
    stats["acts_stats"]["most_common_acts"] = act_counter.most_common(10)
    
    # Average summary length
    if stats["summary_lengths"]:
        stats["avg_summary_length"] = sum(stats["summary_lengths"]) / len(stats["summary_lengths"])
    
    return stats, all_issues


def print_report(stats: Dict, issues: List[str]):
    """Print validation report"""
    
    print("\n" + "="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'─'*80}")
    print(f"Total Cases: {stats['total_cases']}")
    print(f"✅ Valid Cases: {stats['valid_cases']} ({stats['valid_cases']/max(stats['total_cases'],1)*100:.1f}%)")
    print(f"❌ Invalid Cases: {stats['invalid_cases']} ({stats['invalid_cases']/max(stats['total_cases'],1)*100:.1f}%)")
    
    print(f"\n📜 ACTS EXTRACTION")
    print(f"{'─'*80}")
    print(f"Cases with Acts: {stats['acts_stats']['cases_with_acts']} ({stats['acts_stats']['cases_with_acts']/max(stats['total_cases'],1)*100:.1f}%)")
    print(f"Cases without Acts: {stats['acts_stats']['cases_without_acts']} ({stats['acts_stats']['cases_without_acts']/max(stats['total_cases'],1)*100:.1f}%)")
    print(f"Total Acts Extracted: {stats['acts_stats']['total_acts']}")
    print(f"Unique Acts: {stats['acts_stats']['unique_acts_count']}")
    print(f"Avg Acts/Case: {stats['acts_stats']['avg_acts_per_case']:.2f}")
    
    print(f"\n🏆 TOP 10 MOST COMMON ACTS")
    print(f"{'─'*80}")
    for i, (act, count) in enumerate(stats['acts_stats']['most_common_acts'], 1):
        print(f"{i:2}. {act:50} {count:4} cases")
    
    print(f"\n⚖️  OUTCOME DISTRIBUTION")
    print(f"{'─'*80}")
    for outcome, count in stats['outcome_distribution'].most_common():
        pct = count / max(stats['total_cases'], 1) * 100
        print(f"{outcome:20} {count:4} cases ({pct:5.1f}%)")
    
    print(f"\n🏛️  COURT DISTRIBUTION")
    print(f"{'─'*80}")
    for court, count in stats['court_distribution'].most_common():
        pct = count / max(stats['total_cases'], 1) * 100
        print(f"{court:30} {count:4} cases ({pct:5.1f}%)")
    
    print(f"\n📅 YEAR DISTRIBUTION")
    print(f"{'─'*80}")
    for year, count in sorted(stats['year_distribution'].items()):
        pct = count / max(stats['total_cases'], 1) * 100
        print(f"{year:10} {count:4} cases ({pct:5.1f}%)")
    
    print(f"\n📝 SUMMARY STATISTICS")
    print(f"{'─'*80}")
    print(f"Average Length: {stats.get('avg_summary_length', 0):.0f} characters")
    print(f"Min Length: {min(stats['summary_lengths']) if stats['summary_lengths'] else 0} characters")
    print(f"Max Length: {max(stats['summary_lengths']) if stats['summary_lengths'] else 0} characters")
    
    if issues:
        print(f"\n⚠️  VALIDATION ISSUES ({len(issues)})")
        print(f"{'─'*80}")
        for issue in issues[:20]:  # Show first 20
            print(f"  • {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues)-20} more issues")
    else:
        print(f"\n✅ NO VALIDATION ISSUES FOUND")
    
    print("\n" + "="*80)
    
    # Overall grade
    if stats['invalid_cases'] == 0 and stats['acts_stats']['cases_without_acts'] < stats['total_cases'] * 0.1:
        print("GRADE: ✅ EXCELLENT - All data properly structured")
    elif stats['invalid_cases'] < stats['total_cases'] * 0.1:
        print("GRADE: ⚠️  GOOD - Minor issues, mostly complete")
    elif stats['invalid_cases'] < stats['total_cases'] * 0.3:
        print("GRADE: ⚠️  FAIR - Some data quality issues")
    else:
        print("GRADE: ❌ POOR - Significant data quality issues")
    
    print("="*80)


def sample_cases(cases: List[Dict], n: int = 3):
    """Show sample cases"""
    print(f"\n📋 SAMPLE CASES (showing {min(n, len(cases))})")
    print("="*80)
    
    for i, case in enumerate(cases[:n], 1):
        print(f"\nCase {i}:")
        print(f"  Case ID: {case.get('case_id')}")
        print(f"  Court: {case.get('court')}")
        print(f"  Date: {case.get('date')}")
        print(f"  Petitioners: {case.get('petitioners')}")
        print(f"  Respondents: {case.get('respondents')}")
        print(f"  Judges: {case.get('judges')}")
        print(f"  Acts: {case.get('acts_referred')}")
        print(f"  Outcome: {case.get('predicted_outcome')}")
        print(f"  Summary: {case.get('summary', '')[:150]}...")
        print(f"  Source: {case.get('source')}")
        print("─"*80)


def main():
    """Main validation function"""
    
    judgments_file = "output/judgments.json"
    
    try:
        with open(judgments_file, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        print(f"✅ Loaded {len(cases)} cases from {judgments_file}")
        
        # Analyze
        stats, issues = analyze_dataset(cases)
        
        # Print report
        print_report(stats, issues)
        
        # Show samples
        sample_cases(cases, n=3)
        
        # Save report
        report = {
            "validation_timestamp": json.dumps({"timestamp": "now"}),
            "statistics": {
                k: v for k, v in stats.items() 
                if k not in ['acts_stats']  # Skip complex objects
            },
            "issues_count": len(issues),
            "sample_issues": issues[:10]
        }
        
        with open("output/validation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: output/validation_report.json")
        
    except FileNotFoundError:
        print(f"❌ File not found: {judgments_file}")
        print("   Run the download pipeline first: python api.py")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {judgments_file}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()