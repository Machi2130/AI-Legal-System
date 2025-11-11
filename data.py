"""
process_local_pdfs.py
=====================
COMPLETE FIXED VERSION
Process existing PDFs from data/delhi_cases and data/madras_cases
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from groq import Groq
import PyPDF2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LocalPDFProcessor:
    """
    Process local PDFs with:
    - 2 API keys (round-robin)
    - Primary model: Llama 3.3 70B
    - Fallback model: Llama 3.1 8B
    - Resume capability
    """
    
    def __init__(self):
        """Initialize processor"""
        
        # Load API keys (FIXED to detect LLAMA33 format)
        self.api_keys = self._load_api_keys()
        
        if not self.api_keys:
            raise ValueError(
                "❌ No API keys found!\n\n"
                "Set environment variables:\n"
                "  $env:GROQ_API_KEY_LLAMA33_1 = 'gsk_...'\n"
                "  $env:GROQ_API_KEY_LLAMA33_2 = 'gsk_...'\n\n"
                "OR:\n"
                "  $env:GROQ_API_KEY_1 = 'gsk_...'\n"
                "  $env:GROQ_API_KEY_2 = 'gsk_...'\n\n"
                "OR:\n"
                "  $env:GROQ_API_KEY = 'gsk_...'"
            )
        
        # Initialize Groq clients
        self.clients = [Groq(api_key=key) for key in self.api_keys]
        
        # Models
        self.primary_model = "llama-3.3-70b-versatile"
        self.fallback_model = "openai/gpt-oss-120b"
        
        # Round-robin
        self.current_idx = 0
        
        # Output
        self.output_file = "output/judgments.json"
        
        # Stats
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'primary_model': 0,
            'fallback_model': 0,
            'per_key': {i: 0 for i in range(len(self.api_keys))}
        }
        
        print(f"\n{'='*70}")
        print(f"🚀 LOCAL PDF PROCESSOR INITIALIZED")
        print(f"{'='*70}")
        print(f"Primary Model: {self.primary_model}")
        print(f"Fallback Model: {self.fallback_model}")
        print(f"API Keys: {len(self.api_keys)}")
        print(f"Capacity: {30 * len(self.api_keys)} req/min")
        print(f"Output: {self.output_file}")
        print(f"{'='*70}\n")
    
    def _load_api_keys(self) -> List[str]:
        """
        FIXED: Load API keys from environment
        
        Priority:
        1. GROQ_API_KEY_LLAMA33_1, GROQ_API_KEY_LLAMA33_2, ...
        2. GROQ_API_KEY_1, GROQ_API_KEY_2, ...
        3. GROQ_API_KEY (single key)
        """
        keys = []
        
        print("🔍 Searching for API keys...")
        
        # Priority 1: LLAMA33 format (your format)
        for i in range(1, 6):
            key = os.getenv(f"GROQ_API_KEY_LLAMA33_{i}")
            if key and key not in keys:
                keys.append(key)
                masked = f"{key[:8]}...{key[-4:]}"
                print(f"   ✅ Found GROQ_API_KEY_LLAMA33_{i}: {masked}")
        
        # Priority 2: Generic numbered format
        if not keys:
            for i in range(1, 6):
                key = os.getenv(f"GROQ_API_KEY_{i}")
                if key and key not in keys:
                    keys.append(key)
                    masked = f"{key[:8]}...{key[-4:]}"
                    print(f"   ✅ Found GROQ_API_KEY_{i}: {masked}")
        
        # Priority 3: Single key
        if not keys:
            key = os.getenv("GROQ_API_KEY")
            if key:
                keys.append(key)
                masked = f"{key[:8]}...{key[-4:]}"
                print(f"   ✅ Found GROQ_API_KEY: {masked}")
        
        if not keys:
            print("   ❌ No keys found")
        else:
            print(f"   📊 Total: {len(keys)} key(s) loaded\n")
        
        return keys
    
    def _get_next_client(self):
        """Get next client (round-robin)"""
        idx = self.current_idx
        client = self.clients[idx]
        self.current_idx = (idx + 1) % len(self.clients)
        return client, idx
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"      ❌ PDF error: {str(e)[:80]}")
            return ""
    
    def extract_with_groq(self, pdf_text: str, case_id: str, court: str, use_fallback: bool = False) -> Dict:
        """Extract using Groq API"""
        
        if not pdf_text or len(pdf_text) < 100:
            return None
        
        # Truncate
        max_chars = 15000 if not use_fallback else 12000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars]
        
        # Model
        model = self.fallback_model if use_fallback else self.primary_model
        
        # Prompt
        prompt = f"""Analyze this {court} judgment. Return ONLY valid JSON.

JSON SCHEMA:
{{
  "case_id": "{case_id}",
  "court": "{court}",
  "date": "YYYY-MM-DD",
  "petitioners": ["Names"],
  "respondents": ["Names"],
  "judges": ["Names"],
  "case_type": "Type",
  "acts_referred": ["IPC 302", "..."],
  "sections": ["302", "..."],
  "predicted_outcome": "Allowed/Dismissed/Partly Allowed/Disposed",
  "primary_legal_issue": "Core question (1 sentence)",
  "summary": "250-word summary",
  "facts": "Facts",
  "arguments": "Arguments",
  "judgment_reasoning": "Reasoning",
  "key_legal_points": ["Point 1", "..."],
  "legal_issues": ["Issue 1", "..."],
  "precedents_cited": ["Citations"],
  "citations": ["Citations"]
}}

JUDGMENT:
{pdf_text}

JSON:"""
        
        # Retry
        for attempt in range(2):
            try:
                client, key_idx = self._get_next_client()
                self.stats['per_key'][key_idx] += 1
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Extract legal data as JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=3500 if not use_fallback else 2500,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                
                extracted = json.loads(content)
                extracted = self._post_process(extracted, pdf_text, model, key_idx)
                
                if use_fallback:
                    self.stats['fallback_model'] += 1
                else:
                    self.stats['primary_model'] += 1
                
                return extracted
            
            except Exception as e:
                if "429" in str(e) and attempt < 1:
                    time.sleep(3)
                    continue
                if attempt < 1:
                    time.sleep(1)
                    continue
                return None
        
        return None
    
    def _post_process(self, extracted: Dict, pdf_text: str, model: str, key_idx: int) -> Dict:
        """Validate extracted data"""
        
        if not extracted.get('primary_legal_issue'):
            if extracted.get('legal_issues'):
                extracted['primary_legal_issue'] = extracted['legal_issues']
            else:
                extracted['primary_legal_issue'] = "Legal dispute"
        
        list_fields = [
            'petitioners', 'respondents', 'judges', 'acts_referred',
            'sections', 'key_legal_points', 'legal_issues',
            'precedents_cited', 'citations'
        ]
        for field in list_fields:
            if field not in extracted:
                extracted[field] = []
        
        string_fields = [
            'case_id', 'court', 'date', 'case_type', 'predicted_outcome',
            'summary', 'facts', 'arguments', 'judgment_reasoning'
        ]
        for field in string_fields:
            if field not in extracted:
                extracted[field] = "Unknown"
        
        extracted['full_judgment_text'] = pdf_text
        extracted['text_length'] = len(pdf_text)
        extracted['extraction_model'] = model
        extracted['extraction_timestamp'] = datetime.now().isoformat()
        extracted['api_key_index'] = key_idx + 1
        
        return extracted
    
    def process_pdf(self, pdf_path: str, court: str) -> Dict:
        """Process single PDF with fallback"""
        
        case_id = Path(pdf_path).stem
        
        print(f"      📄 Reading PDF...")
        pdf_text = self.extract_text_from_pdf(pdf_path)
        
        if not pdf_text:
            return None
        
        print(f"      📝 {len(pdf_text)} chars")
        print(f"      🔷 Trying {self.primary_model}...")
        
        result = self.extract_with_groq(pdf_text, case_id, court, False)
        
        if result:
            print(f"      ✅ Success with primary")
            result['local_pdf_path'] = pdf_path
            return result
        
        print(f"      🔶 Trying fallback...")
        result = self.extract_with_groq(pdf_text, case_id, court, True)
        
        if result:
            print(f"      ✅ Success with fallback")
            result['local_pdf_path'] = pdf_path
            return result
        
        print(f"      ❌ Both failed")
        return None
    
    def load_existing_cases(self) -> Dict[str, Dict]:
        """Load existing cases"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    cases = json.load(f)
                    return {case['case_id']: case for case in cases}
            except:
                return {}
        return {}
    
    def save_cases(self, cases: List[Dict]):
        """Save cases to JSON"""
        os.makedirs("output", exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved {len(cases)} cases")
    
    def process_folder(self, folder_path: str, court: str):
        """Process all PDFs in folder"""
        
        print(f"\n{'='*70}")
        print(f"📁 {folder_path}")
        print(f"{'='*70}")
        
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        print(f"Found: {len(pdf_files)} PDFs")
        
        existing_cases = self.load_existing_cases()
        print(f"Existing: {len(existing_cases)} cases")
        
        new_cases = []
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            case_id = pdf_path.stem
            
            print(f"\n   [{idx}/{len(pdf_files)}] {case_id}")
            
            if case_id in existing_cases:
                print(f"      ⏭️  Already processed")
                self.stats['skipped'] += 1
                continue
            
            self.stats['total'] += 1
            result = self.process_pdf(str(pdf_path), court)
            
            if result:
                new_cases.append(result)
                existing_cases[case_id] = result
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
            
            time.sleep(2)
            
            if len(new_cases) % 10 == 0 and new_cases:
                self.save_cases(list(existing_cases.values()))
                print(f"\n      💾 Checkpoint saved")
        
        if new_cases:
            self.save_cases(list(existing_cases.values()))
    
    def process_all_folders(self):
        """Process both folders"""
        
        folders = [
            ("data/delhi_cases", "Delhi High Court"),
            ("data/madras_cases", "Madras High Court")
        ]
        
        for folder_path, court in folders:
            if os.path.exists(folder_path):
                self.process_folder(folder_path, court)
            else:
                print(f"\n⚠️  Not found: {folder_path}")
        
        self.print_stats()
    
    def print_stats(self):
        """Print stats"""
        print(f"\n{'='*70}")
        print(f"📊 STATISTICS")
        print(f"{'='*70}")
        print(f"Total: {self.stats['total']}")
        print(f"Success: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped: {self.stats['skipped']}")
        print(f"\nModels:")
        print(f"  Primary (70B): {self.stats['primary_model']}")
        print(f"  Fallback (8B): {self.stats['fallback_model']}")
        print(f"\nAPI Keys:")
        for idx, count in self.stats['per_key'].items():
            print(f"  Key {idx+1}: {count} requests")
        print(f"{'='*70}")


if __name__ == "__main__":
    try:
        processor = LocalPDFProcessor()
        processor.process_all_folders()
        print(f"\n✅ COMPLETE!")
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
