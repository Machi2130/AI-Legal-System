
import json
import pickle
import os
import re
import numpy as np
import torch
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# HELPER FUNCTION - Recursively flatten nested lists/dicts
def flatten_field(value):
  
    if value is None or value == '':
        return ''
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, list):
        parts = []
        for item in value:
            flattened = flatten_field(item)
            if flattened and flattened.strip():
                parts.append(flattened)
        return ' '.join(parts)
    
    if isinstance(value, dict):
        return ' '.join(str(v) for v in value.values() if v)
    
    return str(value)


class SimilarityCaseFinder:
 
    
    def __init__(self,
                 judgments_file: str = "output/judgments.json",
                 model_name: str = "jinaai/jina-embeddings-v2-base-en",
                 embeddings_cache: str = "output/embeddings_jina.pkl",
                 device: str = None):
        
        self.judgments_file = judgments_file
        self.embeddings_cache = embeddings_cache
        self.cases = []
        self.embeddings = None
        
        # Auto-detect GPU with memory check
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🎮 GPU detected: {gpu_name}")
                print(f"💾 VRAM: {gpu_memory:.1f} GB")
                
                if gpu_memory < 10:
                    print(f"⚠️ WARNING: Only {gpu_memory:.1f} GB VRAM - using conservative settings")
            else:
                self.device = 'cpu'
                print(f"⚠️ No GPU detected, using CPU (will be slower)")
        else:
            self.device = device
        
        print(f"🤖 Loading embedding model: {model_name}")
        print(f"   Device: {self.device.upper()}")
        print(f"   Context window: 8192 tokens")
        
        try:
            # Load model on specified device
            self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
            print(f"✅ Model loaded successfully on {self.device.upper()}")
            
            if self.device == 'cuda':
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"   Model VRAM usage: {allocated:.2f} GB")
                
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            print(f"   Run: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
            raise
        
        self.load_cases()
        self.load_or_compute_embeddings()
    
    def load_cases(self):
        """Load cases from JSON"""
        try:
            with open(self.judgments_file, 'r', encoding='utf-8') as f:
                self.cases = json.load(f)
            print(f"✅ Loaded {len(self.cases)} cases")
        except FileNotFoundError:
            print(f"❌ No judgments file found: {self.judgments_file}")
            self.cases = []
        except Exception as e:
            print(f"❌ Error loading cases: {str(e)}")
            self.cases = []
    
    def normalize_legal_terms(self, text: str) -> str:
      
        if not text:
            return ""
        
        # Remove dots from abbreviations
        text = re.sub(r'\b([A-Z])\.\s*([A-Z])\.\s*([A-Z])', r'\1\2\3', text)
        text = re.sub(r'\b([A-Z])\.\s*([A-Z])', r'\1\2', text)
        
        # Normalize "Section X" to just "X"
        text = re.sub(r'\bSection\s+(\d+[A-Z]?)\b', r'\1', text, flags=re.IGNORECASE)
        
        # Normalize common acts
        text = re.sub(r'\bIndian Penal Code\b', 'IPC', text, flags=re.IGNORECASE)
        text = re.sub(r'\bCode of Criminal Procedure\b', 'CrPC', text, flags=re.IGNORECASE)
        text = re.sub(r'\bIncome Tax Act\b', 'IT Act', text, flags=re.IGNORECASE)
        
        return text
    
    def prepare_text_for_embedding(self, case: Dict) -> str:
       
        if not case or not isinstance(case, dict):
            return "Empty case"
        
        parts = []
        
        # PRIORITY 1: Legal issues (4x weight)
        legal_issues = case.get('legal_issues', '') or case.get('legalissues', '')
        if legal_issues:
            normalized = self.normalize_legal_terms(flatten_field(legal_issues)[:800])
            if normalized.strip():
                parts.extend([normalized] * 4)
        
        # PRIORITY 2: Primary legal issue (4x weight)
        primary_issue = case.get('primary_legal_issue', '') or case.get('primarylegalissue', '')
        if primary_issue and primary_issue not in ['Unknown', 'Unknown legal issue']:
            normalized = self.normalize_legal_terms(flatten_field(primary_issue)[:400])
            if normalized.strip():
                parts.extend([normalized] * 4)
        
        # PRIORITY 3: Summary (3x weight)
        summary = case.get('summary', '')
        if summary and summary != 'Unknown':
            normalized = self.normalize_legal_terms(flatten_field(summary)[:1000])
            if normalized.strip():
                parts.extend([normalized] * 3)
        
        # PRIORITY 4: Key legal points (3x weight)
        legal_points = case.get('key_legal_points', '') or case.get('keylegalpoints', '')
        if legal_points:
            normalized = self.normalize_legal_terms(flatten_field(legal_points)[:800])
            if normalized.strip():
                parts.extend([normalized] * 3)
        
        # PRIORITY 5: Facts (2x weight)
        facts = case.get('facts', '')
        if facts and facts != 'Unknown':
            normalized = self.normalize_legal_terms(flatten_field(facts)[:600])
            if normalized.strip():
                parts.extend([normalized] * 2)
        
        # PRIORITY 6: Judgment reasoning (2x weight)
        reasoning = case.get('judgment_reasoning', '') or case.get('judgmentreasoning', '')
        if reasoning and reasoning != 'Unknown':
            normalized = self.normalize_legal_terms(flatten_field(reasoning)[:600])
            if normalized.strip():
                parts.extend([normalized] * 2)
        
        # PRIORITY 7: Acts referred (2x weight)
        acts = case.get('acts_referred', '')
        if acts:
            normalized = self.normalize_legal_terms(flatten_field(acts)[:400])
            if normalized.strip():
                parts.extend([normalized] * 2)
        
        # PRIORITY 8: Arguments (1x)
        arguments = case.get('arguments', '')
        if arguments and arguments != 'Unknown':
            normalized = self.normalize_legal_terms(flatten_field(arguments)[:500])
            if normalized.strip():
                parts.append(normalized)
        
        # PRIORITY 9: Precedents (1x)
        precedents = case.get('precedents_cited', '')
        if precedents:
            normalized = self.normalize_legal_terms(flatten_field(precedents)[:400])
            if normalized.strip():
                parts.append(normalized)
        
        # PRIORITY 10: Metadata
        case_type = case.get('case_type', '')
        if case_type and case_type != 'Unknown':
            parts.append(f"Type: {case_type}")
        
        outcome = case.get('predicted_outcome', '')
        if outcome:
            parts.append(f"Outcome: {outcome}")
        
        court = case.get('court', '')
        if court:
            parts.append(f"Court: {court}")
        
        # Combine all parts
        combined = " ".join(parts)
        
        # CRITICAL: Reduced from 30000 to 10000 chars (saves ~70% VRAM)
        if len(combined) > 10000:
            combined = combined[:10000]
        
        if not combined or not combined.strip():
            return f"Case {case.get('case_id', 'unknown')} - No extractable text"
        
        return combined
    
    def compute_embeddings(self):
        
        try:
            if not self.cases:
                print("⚠️ No cases to embed")
                return
            
            print(f"\n🤖 Computing embeddings for {len(self.cases)} cases...")
            print(f"   Using weighted text preparation (memory-optimized)...")
            
            # Prepare texts
            texts = []
            for idx, case in enumerate(self.cases):
                if (idx + 1) % 10 == 0:
                    print(f"   Preparing text {idx+1}/{len(self.cases)}...")
                text = self.prepare_text_for_embedding(case)
                texts.append(text)
            
            # Validate texts
            texts = [t if t and t.strip() else "Empty case" for t in texts]
            
            print(f"✅ Prepared {len(texts)} texts")
            text_lengths = [len(t) for t in texts]
            avg_length = sum(text_lengths) // len(text_lengths)
            print(f"   Text length: min={min(text_lengths)}, max={max(text_lengths)}, avg={avg_length} chars")
            
            # Conservative batch size for 8GB VRAM
            if self.device == 'cuda':
                # Enable PyTorch memory optimization
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # Use small batch size for safety
                batch_size = 4  # Safe for 8GB with 10KB texts
                
                print(f"🎮 GPU acceleration enabled (8GB VRAM mode)")
                print(f"   Batch size: {batch_size} (conservative for stability)")
                print(f"🚀 Generating embeddings (will take ~2-4 minutes)...")
                
                # Clear all GPU memory before starting
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Show initial VRAM
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"   Starting VRAM: {allocated:.2f} GB")
            else:
                batch_size = 4
                print(f"🤖 Generating embeddings (CPU mode - will take 10-15 minutes)...")
            
            # Encode with progress bar
            self.embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
            
            print(f"✅ Generated embeddings: shape {self.embeddings.shape}")
            
            if self.device == 'cuda':
                # Show final GPU memory usage
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"   Peak VRAM: {reserved:.2f} GB (allocated: {allocated:.2f} GB)")
                
                # Aggressive cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"   Cleared GPU cache")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ GPU OUT OF MEMORY!")
            if self.device == 'cuda':
                try:
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    print(f"   VRAM: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")
                except:
                    pass
                print(f"   Batch size: {batch_size}")
                print(f"\n💡 SOLUTIONS:")
                print(f"   1. Your texts are still too long - reduce limits further in prepare_text_for_embedding")
                print(f"   2. Use CPU mode: Change device='cpu' in __init__")
                print(f"   3. Close other GPU applications (Chrome, games, etc.)")
            
            torch.cuda.empty_cache()
            self.embeddings = None
            raise
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            self.embeddings = None
            raise
    
    def load_or_compute_embeddings(self):
        """Load cached embeddings or compute new ones"""
        if os.path.exists(self.embeddings_cache):
            try:
                print(f"📂 Loading cached embeddings from {self.embeddings_cache}...")
                with open(self.embeddings_cache, 'rb') as f:
                    cache = pickle.load(f)
                self.embeddings = cache['embeddings']
                print(f"✅ Loaded cached embeddings: shape {self.embeddings.shape}")
                return
            except Exception as e:
                print(f"⚠️ Failed to load cache: {str(e)}")
                print(f"   Recomputing embeddings...")
        else:
            print(f"📝 No cache found, computing embeddings...")
        
        self.compute_embeddings()
        self.save_embeddings()
    
    def save_embeddings(self):
        """Save embeddings to cache"""
        if self.embeddings is None:
            return
        
        try:
            os.makedirs("output", exist_ok=True)
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'model_name': 'jinaai/jina-embeddings-v2-base-en',
                    'embedding_dim': self.embeddings.shape[1],
                    'num_cases': len(self.cases),
                    'device': self.device
                }, f)
            print(f"💾 Saved embeddings to {self.embeddings_cache}")
        except Exception as e:
            print(f"⚠️ Failed to save embeddings: {str(e)}")
    
    def find_similar(self, query_case: Dict, topk: int = 5, exclude_self: bool = True) -> List[Tuple[Dict, float]]:
        """Find similar cases to a query case (GPU-accelerated)"""
        try:
            if self.embeddings is None:
                print("❌ No embeddings available")
                return []
            
            # Prepare query text
            query_text = self.prepare_text_for_embedding(query_case)
            
            # Encode query
            query_embedding = self.model.encode(
                [query_text], 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                device=self.device
            )
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k
            top_indices = np.argsort(similarities)[::-1]
            
            # Exclude self if requested
            if exclude_self:
                query_id = query_case.get('case_id')
                top_indices = [idx for idx in top_indices 
                             if self.cases[idx].get('case_id') != query_id]
            
            top_indices = top_indices[:topk]
            results = [(self.cases[idx], float(similarities[idx])) 
                      for idx in top_indices]
            
            return results
            
        except Exception as e:
            print(f"❌ ERROR in find_similar: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def find_similar_by_text(self, query_text: str, topk: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find similar cases by text query (GPU-accelerated)
        Enhanced with normalization and error handling
        """
        try:
            if self.embeddings is None:
                raise ValueError("No embeddings available. Please compute embeddings first.")
            
            if not query_text or not query_text.strip():
                raise ValueError("Query text cannot be empty")
            
            # Normalize query
            normalized_query = self.normalize_legal_terms(query_text)
            
            # Encode query (GPU-accelerated)
            query_embedding = self.model.encode(
                [normalized_query], 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                device=self.device
            )
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:topk]
            results = [(self.cases[idx], float(similarities[idx])) 
                      for idx in top_indices]
            
            return results
            
        except Exception as e:
            print(f"❌ ERROR in find_similar_by_text: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Test the memory-optimized similarity engine
    print("="*70)
    print("Testing Memory-Optimized Similarity Engine (RTX 5080 8GB)")
    print("="*70)
    
    finder = SimilarityCaseFinder()
    print(f"\n✅ Similarity engine ready with {len(finder.cases)} cases")
    
    # Test query
    if finder.cases:
        test_query = "Murder case under IPC Section 302"
        print(f"\n🔍 Testing query: '{test_query}'")
        
        import time
        start = time.time()
        results = finder.find_similar_by_text(test_query, topk=3)
        elapsed = time.time() - start
        
        print(f"\n📊 Top 3 similar cases (found in {elapsed:.3f}s):")
        for idx, (case, score) in enumerate(results, 1):
            print(f"{idx}. {case.get('case_id', 'Unknown')} - Similarity: {score:.2%}")
            print(f"   Issue: {case.get('primary_legal_issue', 'N/A')[:100]}")
