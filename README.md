

---

````markdown
# ⚖️ LegalVault — AI-Powered High Court Judgment Intelligence System

> **Next-Gen Legal Document Intelligence Platform**  
> Transforming unstructured Indian High Court judgments into structured, searchable, and analyzable intelligence.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📘 Overview

**LegalVault** ingests thousands of **Indian High Court judgments** (starting with *Delhi* and *Madras*), automatically extracts entities using **Groq Llama 3.3 70B**, and powers a **semantic search engine** built with **Jina Embeddings v2** and **PyTorch GPU acceleration**.  
The result: instant, meaning-based legal retrieval and analytics for judges, lawyers, and researchers.

---

## 🧩 System Architecture

### 1️⃣ Full Data Pipeline
![System Architecture](diagram-export-11-11-2025-10_50_00.jpg)

> **Flow:** AWS S3 → PDF Text Extraction → LLM Entity Extraction → JSON DB → Semantic & Keyword Search → Analytics & REST API.

### 2️⃣ Similarity Engine Internals
![Similarity Engine](diagram-export-11-11-2025-10_49_23.jpg)

> GPU-optimized similarity computation using **weighted embeddings**, **legal term normalization**, and **persistent caching**.

---

## 🌟 Key Features

| Category | Capability |
|-----------|-------------|
| ⚙️ **Automation** | Fetches & parses judgments from AWS S3 automatically |
| 🧠 **LLM Entity Extraction** | 20+ structured entities (judge names, acts, outcomes, issues) |
| 🔍 **Semantic Search** | Finds legally similar cases via contextual embeddings |
| ⚡ **GPU-Accelerated** | PyTorch 2.9 + CUDA 12.8 (optimized for RTX 50-series) |
| 📊 **Legal Analytics** | Judge stats, act citations, outcome distribution |
| 🌐 **REST API** | Flask-based backend for frontend or integrations |

---

## 🧠 Entity Extraction Schema

| Category | Extracted Fields |
|----------|----------------|
| **Metadata** | Case ID, Court, Date, Judges, Type |
| **Parties** | Petitioners, Respondents |
| **Legal Basis** | Acts, Sections, Legal Issues |
| **Content** | Summary (250 words), Facts, Reasoning |
| **Outcome** | Predicted Result (Allowed/Dismissed) |
| **Citations** | Precedents, Related Cases |

---

## 🧮 Semantic Search Example

```python
results = similarity_engine.find_similar_by_text(
    "Murder case under IPC 302 with circumstantial evidence",
    topk=5
)
````

**Output →**

* Finds contextually similar cases (even with different wordings)
* Ranks by cosine similarity
* Returns judge, act, and outcome metadata

---

## ⚙️ Tech Stack

| Layer               | Tool               | Purpose                     |
| ------------------- | ------------------ | --------------------------- |
| 🧠 **LLM**          | Groq Llama 3.3 70B | Legal entity extraction     |
| 🧩 **Embeddings**   | Jina v2 Base       | 8192-token semantic vectors |
| ⚡ **Framework**     | PyTorch 2.9+       | GPU acceleration            |
| 🌐 **Backend**      | Flask 3.0+         | REST API                    |
| ☁️ **Cloud Source** | AWS S3 (boto3)     | Raw PDF repository          |
| 📂 **Storage**      | JSON               | Case database               |
| 📑 **Text Parser**  | PyPDF2             | PDF-to-text conversion      |

---

## 🧭 Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legalvault.git
cd legalvault
```

### 2. Create & Activate Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) GPU-Optimized PyTorch

* **RTX 5080 / 5090:**
  `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`

* **RTX 4090 or earlier:**
  `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

---

## 🧰 Directory Layout

```
legalvault/
├── src/
│   ├── fetch_data.py           # S3 fetcher
│   ├── entity_extractor.py     # Groq LLM pipeline
│   ├── preprocess.py           # Normalization
│   ├── similarity_engine.py    # Jina embeddings
│   ├── search_engine.py        # Keyword search
│   └── analytics.py            # Statistical module
├── output/
│   ├── judgments.json
│   └── embeddings_jina.pkl
├── api.py                      # Flask API
├── main.py                     # CLI runner
└── requirements.txt
```

---

## 🚀 Usage

### 💡 Run the Full Pipeline

```bash
python main.py
```

### 🌐 Launch REST API

```bash
python api.py
```

Access via:
`http://localhost:5000/api/status` → Ready check
`/api/search` → Keyword Search
`/api/similarity` → Semantic Search
`/api/analytics` → Statistics

---

## 📈 Performance Snapshot (RTX 5080 8GB)

| Task              | Time            | Notes           |
| ----------------- | --------------- | --------------- |
| Entity Extraction | 15–30s per case | Groq LLM        |
| Embedding Cache   | 2–4 mins total  | GPU accelerated |
| Similarity Query  | < 100ms         | Cosine search   |
| Peak VRAM Usage   | ~6GB            | Batch size = 4  |

---

## ⚠️ Common Issues & Fixes

| Symptom                  | Cause                   | Fix                       |
| ------------------------ | ----------------------- | ------------------------- |
| ❌ *CUDA not found*       | Wrong PyTorch build     | Use nightly `cu128` wheel |
| ⚠️ *Out of memory*       | Small GPU               | Reduce batch size to 2    |
| 🕒 *Rate limit exceeded* | Groq limit (30 req/min) | Add multiple API keys     |
| ❗ *No PDF found (404)*   | S3 missing file         | Skip case gracefully      |

---

## 🔮 Future Enhancements

* [ ] Expand to Bombay & Calcutta High Courts
* [ ] Integrate Legal-BERT embeddings
* [ ] Add citation network (Neo4j)
* [ ] Real-time scraping updates
* [ ] Hindi & Tamil multilingual support
* [ ] D3.js-based analytics dashboard

---

## 🤝 Contributing

1. Fork → Feature Branch → Pull Request
2. Use `pytest` for validation
3. Format with `black` and lint with `flake8`

```bash
pip install pytest black flake8
pytest
black src/ api.py
flake8 src/
```

---

## 📜 License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🧑‍💻 Author

**Shravani**
GitHub: [@yourusername](https://github.com/yourusername)
Project: [LegalVault](https://github.com/yourusername/legalvault)

---

## 🌟 If this project inspired you, star it!

```
 _                      _  __     __          _ _   
| |    ___  __ _  __ _| | \ \   / /_ _ _   _| | |_ 
| |   / _ \/ _` |/ _` | |  \ \ / / _` | | | | | __|
| |__|  __/ (_| | (_| | |   \ V / (_| | |_| | | |_ 
|_____\___|\__, |\__,_|_|    \_/ \__,_|\__,_|_|\__|
           |___/                                    
```
Video Link
"https://drive.google.com/drive/folders/16lMHriHEiKWp3YjPZfP-IhoxCSNuze7R?usp=sharing"
---

### 🏛️ *Empowering Legal Intelligence through AI.*

---

