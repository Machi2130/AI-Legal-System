"""
src/fetch_data.py
=================
Download PDFs from S3 for Delhi and Madras High Courts
"""

import boto3
import os
import json
import tempfile
import PyPDF2
import time
from botocore.config import Config
from botocore import UNSIGNED
from typing import List, Dict
import re


class DataFetcher:
    COURTS = {
        'delhi': {
            'code': '7_26',
            'name': 'Delhi High Court',
            'folder': 'data/delhi_cases'
        },
        'madras': {
            'code': '33_10',
            'name': 'Madras High Court',
            'folder': 'data/madras_cases'
        }
    }
    
    def __init__(self):
        self.s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self.bucket = "indian-high-court-judgments"
        
        # Create data folders
        for court_info in self.COURTS.values():
            os.makedirs(court_info['folder'], exist_ok=True)
        
        print("✅ DataFetcher initialized")
    
    def extract_bench_from_path(self, json_key: str) -> str:
        """Extract bench name from JSON metadata path"""
        bench_match = re.search(r'bench=([^/]+)', json_key)
        return bench_match.group(1) if bench_match else None
    
    def construct_pdf_path(self, json_key: str, year: str, court_code: str) -> str:
        """Construct correct S3 PDF path"""
        bench = self.extract_bench_from_path(json_key)
        if not bench:
            return None
        
        filename = os.path.basename(json_key).replace('.json', '.pdf')
        return f"data/pdf/year={year}/court={court_code}/bench={bench}/{filename}"
    
    def download_pdf(self, pdf_s3_path: str, local_path: str) -> str:
        """Download PDF and extract text"""
        tmp_path = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()
            
            # Download from S3
            self.s3.download_file(self.bucket, pdf_s3_path, tmp_path)
            
            # Extract text
            text_parts = []
            with open(tmp_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except:
                        continue
            
            full_text = "\n\n".join(text_parts)
            
            # Save PDF to local folder
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f_out:
                with open(tmp_path, 'rb') as f_in:
                    f_out.write(f_in.read())
            
            return full_text
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    time.sleep(0.1)
                    os.unlink(tmp_path)
                except:
                    pass
    
    def fetch_court_data(self, court_key: str, year: str = "2023", max_files: int = 20) -> List[Dict]:
        if court_key not in self.COURTS:
            print(f"❌ Invalid court key: {court_key}")
            return []
        
        court_info = self.COURTS[court_key]
        court_code = court_info['code']
        court_name = court_info['name']
        local_folder = court_info['folder']
        
        print(f"\n{'='*70}")
        print(f"🏛️  {court_name} (Year {year})")
        print(f"{'='*70}")
        
        # List JSON metadata
        metadata_prefix = f"metadata/json/year={year}/court={court_code}/"
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=metadata_prefix,
                MaxKeys=max_files * 2
            )
            
            json_files = [
                obj["Key"] for obj in response.get("Contents", [])
                if obj["Key"].endswith(".json")
            ][:max_files]
            
            print(f"📂 Found {len(json_files)} metadata files")
            
            fetched_data = []
            for idx, json_key in enumerate(json_files, 1):
                case_id = os.path.basename(json_key).replace('.json', '')
                print(f"\n  [{idx}/{len(json_files)}] {case_id}")
                
                # Construct PDF path
                pdf_s3_path = self.construct_pdf_path(json_key, year, court_code)
                if not pdf_s3_path:
                    print(f"   ⚠️  Could not construct PDF path")
                    continue
                
                # Verify PDF exists
                try:
                    self.s3.head_object(Bucket=self.bucket, Key=pdf_s3_path)
                except:
                    print(f"   ❌ PDF not found")
                    continue
                
                # Download PDF
                print(f"   📥 Downloading...")
                local_pdf_path = os.path.join(local_folder, f"{case_id}.pdf")
                pdf_text = self.download_pdf(pdf_s3_path, local_pdf_path)
                
                if not pdf_text or len(pdf_text) < 100:
                    print(f"   ⚠️  Insufficient text")
                    continue
                
                print(f"   ✅ Downloaded {len(pdf_text)} chars")
                
                # Store data
                fetched_data.append({
                    'case_id': case_id,
                    'court': court_name,
                    'pdf_text': pdf_text,
                    's3_path': pdf_s3_path,
                    'local_path': local_pdf_path,
                    'text_length': len(pdf_text)
                })
                
                # Rate limiting
                time.sleep(2)
            
            print(f"\n✅ Fetched {len(fetched_data)} PDFs for {court_name}")
            return fetched_data
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def download_judgments(self, year: int = 2023, max_files_per_court: int = 20) -> List[Dict]:
        """
        API-compatible wrapper to download from both courts
        
        Parameters:
        - year: Year to download (2023, 2022, 2021, etc.)
        - max_files_per_court: Max PDFs per court
        
        Returns:
        - Combined list of all cases from both courts
        """
        all_cases = []
        
        print(f"\n{'='*70}")
        print(f"📥 DOWNLOADING JUDGMENTS FOR YEAR {year}")
        print(f"   Max files per court: {max_files_per_court}")
        print(f"{'='*70}\n")
        
        # Fetch from Delhi High Court
        delhi_cases = self.fetch_court_data(
            court_key='delhi',
            year=str(year),
            max_files=max_files_per_court
        )
        all_cases.extend(delhi_cases)
        
        # Fetch from Madras High Court
        madras_cases = self.fetch_court_data(
            court_key='madras',
            year=str(year),
            max_files=max_files_per_court
        )
        all_cases.extend(madras_cases)
        
        print(f"\n{'='*70}")
        print(f"✅ TOTAL CASES DOWNLOADED: {len(all_cases)}")
        print(f"   Delhi: {len(delhi_cases)}, Madras: {len(madras_cases)}")
        print(f"{'='*70}\n")
        
        return all_cases


if __name__ == "__main__":
    # Test
    fetcher = DataFetcher()
    
    # Test the new API-compatible method
    cases = fetcher.download_judgments(year=2023, max_files_per_court=3)
    print(f"\n✅ Total downloaded: {len(cases)} cases")
