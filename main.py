"""
main.py
=======
Main pipeline for Legal Intelligence System
Targets Delhi and Madras High Courts
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, 'src')

from fetch_data import DataFetcher
from entity_extractor import EntityExtractor
from preprocess import DataPreprocessor
from search_engine import CaseSearchEngine
from similarity_engine import SimilarityCaseFinder
from analytics import LegalAnalytics

def run_complete_pipeline(max_cases_per_court: int = 20):

    
    print("\n" + "="*70)
    print("🚀 LEGAL INTELLIGENCE SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    print("\nTargeting: Delhi High Court + Madras High Court")
    print(f"Max cases per court: {max_cases_per_court}")
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ Error: GROQ_API_KEY not set")
        print("PowerShell: $env:GROQ_API_KEY = \"gsk_your_key_here\"")
        return
    
    # STEP 1: Fetch PDFs
    print("\n" + "="*70)
    print("STEP 1: FETCHING PDFs FROM S3")
    print("="*70)
    
    fetcher = DataFetcher()
    
    # Fetch Delhi cases
    delhi_data = fetcher.fetch_court_data('delhi', year='2023', max_files=max_cases_per_court)
    
    # Fetch Madras cases
    madras_data = fetcher.fetch_court_data('madras', year='2023', max_files=max_cases_per_court)
    
    all_fetched = delhi_data + madras_data
    
    print(f"\n✅ Total PDFs fetched: {len(all_fetched)}")
    
    if not all_fetched:
        print("❌ No data fetched. Exiting.")
        return
    
    # STEP 2: Extract entities with Groq
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING ENTITIES WITH GROQ")
    print("="*70)
    
    extractor = EntityExtractor()
    extracted_cases = extractor.extract_batch(all_fetched)
    
    print(f"\n✅ Extracted {len(extracted_cases)} cases")
    
    if not extracted_cases:
        print("❌ No cases extracted. Exiting.")
        return
    
    # STEP 3: Preprocess
    print("\n" + "="*70)
    print("STEP 3: PREPROCESSING DATA")
    print("="*70)
    
    preprocessor = DataPreprocessor()
    processed_cases = preprocessor.process(extracted_cases)
    
    print(f"\n✅ Processed {len(processed_cases)} cases")
    
    # STEP 4: Save to output/judgments.json
    print("\n" + "="*70)
    print("STEP 4: SAVING TO output/judgments.json")
    print("="*70)
    
    os.makedirs("output", exist_ok=True)
    output_file = "output/judgments.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_cases, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(processed_cases)} cases to {output_file}")
    
    # STEP 5: Build similarity index
    print("\n" + "="*70)
    print("STEP 5: BUILDING SIMILARITY INDEX")
    print("="*70)
    
    similarity_finder = SimilarityCaseFinder()
    
    # STEP 6: Generate analytics
    print("\n" + "="*70)
    print("STEP 6: GENERATING ANALYTICS")
    print("="*70)
    
    analytics = LegalAnalytics()
    analytics.print_report()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  - {len(processed_cases)} cases processed")
    print(f"  - Delhi: {len([c for c in processed_cases if 'Delhi' in c.get('court', '')])}")
    print(f"  - Madras: {len([c for c in processed_cases if 'Madras' in c.get('court', '')])}")
    print(f"\nOutput files:")
    print(f"  - output/judgments.json (main output)")
    print(f"  - output/embeddings.pkl (similarity index)")
    print(f"  - data/delhi_cases/ (Delhi PDFs)")
    print(f"  - data/madras_cases/ (Madras PDFs)")


if __name__ == "__main__":
    # Run with 20 cases per court (total 40 cases)
    run_complete_pipeline(max_cases_per_court=20)
