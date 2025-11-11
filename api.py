"""
api.py
======
Flask REST API - FIXED to use existing methods
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from fetch_data import DataFetcher
from entity_extractor import EntityExtractor
from preprocess import DataPreprocessor
from search_engine import CaseSearchEngine
from similarity_engine import SimilarityCaseFinder
from analytics import LegalAnalytics

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
JUDGMENTS_FILE = "output/judgments.json"

# ============================================
# FRONTEND ROUTES
# ============================================

@app.route('/')
def serve_frontend():
    """Serve the main HTML frontend"""
    return send_from_directory('.', 'index.html')

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_search_engine():
    """Get or create search engine instance"""
    if not os.path.exists(JUDGMENTS_FILE):
        return None
    return CaseSearchEngine(JUDGMENTS_FILE)

def get_similarity_engine():
    """Get or create similarity engine instance"""
    if not os.path.exists(JUDGMENTS_FILE):
        return None
    return SimilarityCaseFinder(JUDGMENTS_FILE)

def get_analytics_engine():
    """Get or create analytics engine instance"""
    if not os.path.exists(JUDGMENTS_FILE):
        return None
    return LegalAnalytics(JUDGMENTS_FILE)

# ============================================
# API ROUTES
# ============================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status and statistics"""
    try:
        if not os.path.exists(JUDGMENTS_FILE):
            return jsonify({
                "status": "no_data",
                "total_cases": 0,
                "total_courts": 0,
                "total_judges": 0,
                "total_acts": 0
            })
        
        with open(JUDGMENTS_FILE, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        # Calculate statistics
        courts = set()
        judges = set()
        acts = set()
        
        for case in cases:
            if case.get('court'):
                courts.add(case['court'])
            
            case_judges = case.get('judges', [])
            if isinstance(case_judges, list):
                judges.update([j for j in case_judges if j])
            
            case_acts = case.get('acts_referred', [])
            if isinstance(case_acts, list):
                acts.update([a for a in case_acts if a])
        
        return jsonify({
            "status": "ready",
            "total_cases": len(cases),
            "total_courts": len(courts),
            "total_judges": len(judges),
            "total_acts": len(acts)
        })
        
    except Exception as e:
        print(f"ERROR /api/status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_cases():
    """Search cases using keyword search"""
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return jsonify({"error": "No data available"}), 404
        
        data = request.get_json() or {}
        query = data.get('query', '')
        court = data.get('court', '')
        outcome = data.get('outcome', '')
        case_type = data.get('case_type', '')
        
        # Perform search
        results = search_engine.search(
            keyword=query if query else None,
            court=court if court else None,
            outcome=outcome if outcome else None,
            case_type=case_type if case_type else None
        )
        
        # Limit results
        results = results[:50]
        
        return jsonify({
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        print(f"ERROR /api/search: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/similarity', methods=['POST'])
def find_similar_cases():
    """Find similar cases using semantic search"""
    try:
        similarity_engine = get_similarity_engine()
        if not similarity_engine:
            return jsonify({"error": "No data available"}), 404
        
        data = request.get_json() or {}
        query_text = data.get('query_text', '')
        top_k = data.get('top_k', 5)
        
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        
        # Find similar cases
        results = similarity_engine.find_similar_by_text(query_text, topk=top_k)
        
        # Format results
        formatted_results = [
            {
                "case": case,
                "similarity": float(similarity)
            }
            for case, similarity in results
        ]
        
        return jsonify({
            "results": formatted_results,
            "count": len(formatted_results)
        })
        
    except Exception as e:
        print(f"ERROR /api/similarity: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/<analysis_type>', methods=['GET'])
def get_analytics(analysis_type):
    """Get analytics data"""
    try:
        analytics_engine = get_analytics_engine()
        if not analytics_engine:
            return jsonify({"error": "No data available"}), 404
        
        if analysis_type == 'judges':
            data = analytics_engine.judge_statistics()
        elif analysis_type == 'acts':
            data = analytics_engine.most_cited_acts()
        elif analysis_type == 'courts':
            data = analytics_engine.court_distribution()
        elif analysis_type == 'outcomes':
            data = analytics_engine.outcome_distribution()
        else:
            return jsonify({"error": "Invalid analysis type"}), 400
        
        # Limit to top 20
        data = data[:20] if isinstance(data, list) else data
        
        return jsonify({"data": data})
        
    except Exception as e:
        print(f"ERROR /api/analytics/{analysis_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/download', methods=['POST'])
def download_and_process():
    """Download PDFs and process them - FIXED VERSION"""
    try:
        data = request.get_json() or {}
        year = str(data.get('year', 2023))  # Convert to string
        max_files = data.get('max_files', 10)
        
        print(f"\n{'='*70}")
        print(f"📥 API Download Request")
        print(f"   Year: {year}, Max files: {max_files}")
        print(f"{'='*70}\n")
        
        # Check API key
        if not os.getenv("GROQ_API_KEY"):
            return jsonify({"error": "GROQ_API_KEY not set"}), 500
        
        # STEP 1: Fetch data using EXISTING methods
        print("STEP 1: Fetching PDFs from S3...")
        fetcher = DataFetcher()
        
        delhi_data = fetcher.fetch_court_data('delhi', year=year, max_files=max_files)
        madras_data = fetcher.fetch_court_data('madras', year=year, max_files=max_files)
        cases_data = delhi_data + madras_data
        
        if not cases_data:
            return jsonify({"error": "No cases downloaded"}), 500
        
        print(f"✅ Downloaded {len(cases_data)} PDFs total")
        
        # STEP 2: Extract entities using EXISTING batch method
        print("\nSTEP 2: Extracting entities with LLM...")
        extractor = EntityExtractor()
        extracted_cases = extractor.extract_batch(cases_data)
        
        print(f"✅ Extracted {len(extracted_cases)} cases")
        
        # STEP 3: Preprocess
        print("\nSTEP 3: Preprocessing...")
        preprocessor = DataPreprocessor()
        processed_cases = preprocessor.process(extracted_cases)
        
        print(f"✅ Processed {len(processed_cases)} cases")
        
        # STEP 4: Load existing and merge
        existing_cases = []
        if os.path.exists(JUDGMENTS_FILE):
            with open(JUDGMENTS_FILE, 'r', encoding='utf-8') as f:
                existing_cases = json.load(f)
        
        # Merge (avoid duplicates)
        existing_ids = {case.get('case_id') for case in existing_cases}
        new_cases = [case for case in processed_cases if case.get('case_id') not in existing_ids]
        all_cases = existing_cases + new_cases
        
        # STEP 5: Save
        os.makedirs("output", exist_ok=True)
        with open(JUDGMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_cases, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(all_cases)} total cases to database")
        
        # STEP 6: Delete embeddings cache to force rebuild
        for cache_file in ["output/embeddings.pkl", "output/embeddings_jina.pkl"]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Deleted cache: {cache_file}")
        
        print(f"\n{'='*70}")
        print(f"✅ DOWNLOAD COMPLETE")
        print(f"{'='*70}\n")
        
        return jsonify({
            "status": "success",
            "files_downloaded": len(cases_data),
            "cases_processed": len(processed_cases),
            "cases_added": len(new_cases),
            "total_cases": len(all_cases)
        })
        
    except Exception as e:
        print(f"\nERROR /api/download: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/cases', methods=['GET'])
def debug_cases():
    """Debug: Show first 3 cases"""
    try:
        if not os.path.exists(JUDGMENTS_FILE):
            return jsonify({
                'total_cases': 0,
                'error': 'No data file found'
            })
        
        with open(JUDGMENTS_FILE, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        return jsonify({
            'total_cases': len(cases),
            'sample_cases': cases[:3] if len(cases) > 0 else [],
            'sample_fields': list(cases[0].keys()) if len(cases) > 0 else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 LegalVault API Server Starting...")
    print("="*70)
    print(f"Frontend: http://localhost:5000")
    print(f"API Base: http://localhost:5000/api")
    print(f"Debug: http://localhost:5000/api/debug/cases")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
