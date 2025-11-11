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