"""
src/entity_extractor.py
=======================
CORRECTED: Using proper Groq model name
"""

import os
import json
import re
import time
from groq import Groq
from datetime import datetime
from typing import Dict, List

class EntityExtractor:
    """
    Llama 3.3 70B Entity Extractor using Groq API
    """
    
    def __init__(self, groq_api_key: str = None):
        # Load API keys
        self.api_keys = self._load_keys(groq_api_key)
        
        if not self.api_keys:
            raise ValueError(
                " No API keys found!"
               
            )
        
        # Initialize clients
        self.clients = [Groq(api_key=key) for key in self.api_keys]
        
        # MODEL NAME 
        self.model = "llama-3.3-70b-versatile"
        
        # Round-robin
        self.current_idx = 0
        
        # Stats
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'per_key': {i: 0 for i in range(len(self.api_keys))}
        }
        
        print(f"\n{'='*70}")
        print(f"✅ EntityExtractor Initialized")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"API Keys: {len(self.api_keys)}")
        print(f"Rate Limits (per key):")
        print(f"  Free Tier: 30 req/min, 12K tokens/min")
        print(f"  With {len(self.api_keys)} keys: {30 * len(self.api_keys)} req/min")
        print(f"{'='*70}\n")
    
    def _load_keys(self, provided_key: str = None) -> List[str]:
        """Load API keys"""
        keys = []
        
        if provided_key:
            keys.append(provided_key)
            return keys
        
        # Try different env vars
        for i in range(1, 6):
            key = os.getenv(f"GROQ_API_KEY_LLAMA33_{i}")
            if key and key not in keys:
                keys.append(key)
        
        if not keys:
            for i in range(1, 6):
                key = os.getenv(f"GROQ_API_KEY_{i}")
                if key and key not in keys:
                    keys.append(key)
        
        if not keys:
            key = os.getenv("GROQ_API_KEY")
            if key:
                keys.append(key)
        
        return keys
    
    def _get_next_client(self):
        """Round-robin client selection"""
        idx = self.current_idx
        client = self.clients[idx]
        self.current_idx = (idx + 1) % len(self.clients)
        return client, idx
    
    def extract(self, pdf_text: str, case_id: str, court: str) -> Dict:
        """Extract using Llama 3.3 70B"""
        
        self.stats['total'] += 1
        
        if not pdf_text or len(pdf_text) < 100:
            self.stats['failed'] += 1
            return None
        
        # Truncate
        max_chars = 15000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars]
        
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
  "acts_referred": ["IPC 302", "CrPC 197", "..."],
  "sections": ["302", "197", "..."],
  "predicted_outcome": "Allowed/Dismissed/Partly Allowed/Disposed",
  "primary_legal_issue": "Core legal question in ONE sentence",
  "summary": "250-word summary",
  "facts": "Detailed facts",
  "arguments": "Key arguments",
  "judgment_reasoning": "Court reasoning",
  "key_legal_points": ["Point 1", "Point 2", "..."],
  "legal_issues": ["Issue 1", "Issue 2", "..."],
  "precedents_cited": ["Case citations"],
  "citations": ["Full citations"]
}}

JUDGMENT:
{pdf_text}

JSON:"""
        
        # Retry with different keys
        max_retries = min(3, len(self.clients))
        
        for attempt in range(max_retries):
            try:
                client, key_idx = self._get_next_client()
                self.stats['per_key'][key_idx] += 1
                
                # Call Groq with CORRECT model name
                response = client.chat.completions.create(
                    model=self.model,  # "llama-3.3-70b-versatile"
                    messages=[
                        {"role": "system", "content": "Extract legal data as JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=3500,
                    response_format={"type": "json_object"}
                )
                
                # Parse
                content = response.choices[0].message.content.strip()
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                
                extracted = json.loads(content)
                
                # Post-process
                extracted = self._post_process(extracted, pdf_text, key_idx)
                
                self.stats['success'] += 1
                return extracted
            
            except Exception as e:
                error_msg = str(e)
                
                # Rate limit handling
                if "429" in error_msg:
                    print(f"      ⚠️  Rate limit (key {key_idx+1})")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                
                print(f"      ⚠️  Error: {error_msg[:100]}")
                
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                
                self.stats['failed'] += 1
                return None
        
        self.stats['failed'] += 1
        return None
    
    def _post_process(self, extracted: Dict, pdf_text: str, key_idx: int) -> Dict:
        """Validate extracted data"""
        
        # Ensure primary_legal_issue
        if not extracted.get('primary_legal_issue'):
            if extracted.get('legal_issues'):
                extracted['primary_legal_issue'] = extracted['legal_issues']
            else:
                extracted['primary_legal_issue'] = "Legal dispute"
        
        # Ensure lists
        list_fields = [
            'petitioners', 'respondents', 'judges', 'acts_referred',
            'sections', 'key_legal_points', 'legal_issues',
            'precedents_cited', 'citations'
        ]
        for field in list_fields:
            if field not in extracted:
                extracted[field] = []
        
        # Ensure strings
        string_fields = [
            'case_id', 'court', 'date', 'case_type', 'predicted_outcome',
            'summary', 'facts', 'arguments', 'judgment_reasoning'
        ]
        for field in string_fields:
            if field not in extracted:
                extracted[field] = "Unknown"
        
        # Metadata
        extracted['full_judgment_text'] = pdf_text
        extracted['text_length'] = len(pdf_text)
        extracted['extraction_model'] = self.model
        extracted['extraction_timestamp'] = datetime.now().isoformat()
        extracted['api_key_index'] = key_idx + 1
        
        return extracted
    
    def extract_batch(self, cases_data: List[Dict]) -> List[Dict]:
        """Batch extraction"""
        
        print(f"\n{'='*70}")
        print(f"🤖 BATCH EXTRACTION")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Cases: {len(cases_data)}")
        print(f"API keys: {len(self.api_keys)}")
        
        extracted_cases = []
        
        for idx, case_data in enumerate(cases_data, 1):
            print(f"\n   [{idx}/{len(cases_data)}] {case_data.get('case_id', 'Unknown')}")
            
            result = self.extract(
                pdf_text=case_data.get('pdf_text', ''),
                case_id=case_data.get('case_id', 'Unknown'),
                court=case_data.get('court', 'Unknown')
            )
            
            if result:
                result['source'] = case_data.get('s3_path', '')
                result['local_pdf'] = case_data.get('local_path', '')
                extracted_cases.append(result)
                print(f"      ✅ Success (key {result['api_key_index']})")
            else:
                print(f"      ❌ Failed")
            
            # Rate limiting: 2 seconds per request (safe for 30 req/min)
            time.sleep(2.0)
        
        self.print_stats()
        print(f"\n✅ Extracted {len(extracted_cases)}/{len(cases_data)} cases")
        return extracted_cases
    
    def print_stats(self):
        """Print stats"""
        print(f"\n{'='*70}")
        print(f"📊 STATISTICS")
        print(f"{'='*70}")
        print(f"Total: {self.stats['total']}")
        print(f"Success: {self.stats['success']} ({self.stats['success']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"Failed: {self.stats['failed']}")
        print(f"\nPer-key:")
        for idx, count in self.stats['per_key'].items():
            print(f"  Key {idx+1}: {count} requests")
        print(f"{'='*70}")


if __name__ == "__main__":
    extractor = EntityExtractor()
    print("✅ Ready!")
