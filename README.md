# ◈ InternIQ — AI-Powered Internship Recommendation System

> ML-driven internship matching using TF-IDF + cosine similarity, FastAPI backend, NLP resume parsing, and a sleek dark-theme frontend.

---

## 📐 Project Architecture

```
User Input (Skills / Interests / Resume PDF)
        │
        ▼
┌────────────────────────────────────────────────┐
│              FastAPI Backend (main.py)         │
│                                                │
│  POST /api/recommend    POST /api/upload_resume│
│         │                       │              │
│         ▼                       ▼              │
│  RecommendationEngine    ResumeParser          │
│  ┌─────────────────┐    ┌──────────────────┐  │
│  │ preprocessor.py │    │ PyMuPDF / pdfminer│  │
│  │ feature_extractor│   │ Skill extraction  │  │
│  │ TF-IDF + cosine │    │ Section parsing   │  │
│  └─────────────────┘    └──────────────────┘  │
│         │                                      │
│         ▼                                      │
│  Ranked Results + Explanations                 │
└────────────────────────────────────────────────┘
        │
        ▼
Frontend (HTML + CSS + JS)
  • Form: skills, interests, filters
  • Drag-and-drop PDF upload
  • Animated score rings per card
  • "Why recommended" accordion
```

---

## 🗂 Folder Structure

```
internship-recommender/
│
├── backend/
│   ├── main.py                   # FastAPI app, all API endpoints
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── preprocessor.py       # Text cleaning, skill extraction, corpus builder
│   │   ├── feature_extractor.py  # TF-IDF vectorizer wrapper + cosine similarity
│   │   ├── recommender.py        # Orchestration: profile building → ranking → explain
│   │   └── resume_parser.py      # PDF parsing, section detection, contact extraction
│   └── data/
│       ├── internships.json      # 15 sample internship records
│       └── tfidf_cache.pkl       # Auto-generated on first run
│
├── frontend/
│   ├── templates/
│   │   └── index.html            # Main HTML (served by FastAPI)
│   └── static/
│       ├── css/
│       │   └── style.css         # Dark editorial theme, card styles, animations
│       └── js/
│           └── app.js            # All frontend logic (upload, API calls, rendering)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10+  
- pip  
- A terminal (PowerShell / bash / zsh)

### 2. Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd internship-recommender

# Or just extract the zip and cd into it
```

### 3. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the server

```bash
cd backend
python main.py
```

The server starts at **http://localhost:8000**

Open your browser and go to **http://localhost:8000** — the frontend loads automatically.

### 6. Optional: hot-reload during development

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🔌 API Reference

### `GET /api/health`
Returns server status.
```json
{ "status": "ok", "engine_ready": true }
```

---

### `GET /api/metadata`
Returns domain and location options for filter dropdowns.
```json
{
  "domains":   ["AI/ML", "Blockchain", "Cloud Computing", "..."],
  "locations": ["Bangalore", "Gurgaon", "Mumbai", "Remote", "..."]
}
```

---

### `POST /api/recommend`
Main recommendation endpoint.

**Request body:**
```json
{
  "skills":      "Python, PyTorch, NLP",
  "interests":   "I love working on deep learning and language models",
  "resume_text": "",
  "location":    "Bangalore",
  "domain":      "AI/ML",
  "top_k":       8
}
```

**Response:**
```json
{
  "count": 3,
  "recommendations": [
    {
      "id": 1,
      "title": "Machine Learning Engineer Intern",
      "company": "DeepMind Labs",
      "location": "Bangalore",
      "domain": "AI/ML",
      "duration": "6 months",
      "stipend": "₹50,000/month",
      "required_skills": "Python, PyTorch, ...",
      "relevance_score": 78.4,
      "match_grade": "Excellent",
      "explanation": "Your skills in **Python, Pytorch, NLP** directly match..."
    }
  ]
}
```

---

### `POST /api/upload_resume`
Upload a PDF résumé. Returns extracted text, skills, contact info, and section previews.

**Form field:** `file` (multipart/form-data, PDF only, max 10 MB)

**Response:**
```json
{
  "filename": "resume.pdf",
  "word_count": 450,
  "extracted_skills": ["python", "machine learning", "pytorch"],
  "contact": { "email": "you@example.com", "phone": "9876543210" },
  "sections": { "skills": "...", "experience": "..." },
  "raw_text_preview": "John Doe\nSoftware Engineer..."
}
```

---

## 🧠 ML Pipeline Explained

### Step 1 — Text Preprocessing (`preprocessor.py`)
- Lowercase, remove URLs and special chars
- Remove stop words (custom lightweight list — no NLTK download needed)
- Normalise skill synonyms (`js → javascript`, `ml → machine learning`)

### Step 2 — Feature Extraction (`feature_extractor.py`)
- **TF-IDF Vectorizer** with bigrams (1,2), 8000 features, sublinear TF scaling
- Internship corpus is vectorised once at startup and cached as a `.pkl` file
- User profile is transformed using the same fitted vocabulary

### Step 3 — Scoring (`recommender.py`)
- **Cosine Similarity** between the user profile vector and each internship vector
- Score is converted to a 0–100 relevance percentage
- Skills are weighted: user skills are repeated 3× in the profile string before vectorisation

### Step 4 — Explanation Generation
- Matches user-provided skills against the internship's required skills using regex
- Identifies top TF-IDF terms that contributed to the match
- Returns a human-readable markdown explanation string

### Step 5 — Filtering
- Optional post-scoring filter by location and/or domain
- `Remote` internships always pass the location filter

---

## 🚀 Future Improvements

### 1. Semantic Search with Sentence Transformers
Replace TF-IDF with dense embeddings for much better semantic understanding:

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
internship_embeddings = model.encode(corpus)          # (n × 384)
user_embedding        = model.encode([user_profile])  # (1 × 384)

from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity(user_embedding, internship_embeddings).flatten()
```
This captures meaning, not just keyword overlap. "Deep learning" would now match "neural networks" and "AI research" without needing synonym lists.

---

### 2. Vector Database with FAISS (for scale)
When the internship dataset grows to thousands of records, use FAISS for sub-millisecond ANN search:

```python
# pip install faiss-cpu
import faiss
import numpy as np

dim    = 384   # embedding dimension
index  = faiss.IndexFlatIP(dim)            # inner product = cosine on normalised vecs
norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
index.add(embeddings / norms)              # add normalised embeddings

query_norm = query_vec / np.linalg.norm(query_vec)
D, I = index.search(query_norm, k=10)     # D = scores, I = indices
```

---

### 3. LLM-Powered Explanation (Generative AI)
Replace the rule-based explanation with an LLM call for richer, contextual reasoning:

```python
import anthropic

client = anthropic.Anthropic()

def llm_explain(internship, user_skills, user_interests):
    prompt = f"""
    A student with these skills: {user_skills}
    and interests: {user_interests}
    
    was matched to this internship:
    Title: {internship['title']}
    Company: {internship['company']}
    Required skills: {internship['required_skills']}
    
    In 2-3 sentences, explain why this is a good match.
    Be specific about which skills align and what the student will learn.
    """
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text
```

---

### 4. Collaborative Filtering
Track which internships users click/apply to and build a user–internship interaction matrix. Use matrix factorisation (SVD) to surface hidden preferences:

```python
from sklearn.decomposition import TruncatedSVD
# Build user × internship click matrix from logs
# Factorise → user latent factors + internship latent factors
# Score new internships using dot product of user factor × internship factors
```

---

### 5. Resume OCR (Scanned PDFs)
Handle image-based scanned resumes using `pytesseract`:

```python
import pytesseract
from PIL import Image
import fitz

doc = fitz.open(stream=pdf_bytes, filetype="pdf")
for page in doc:
    pix  = page.get_pixmap(dpi=200)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
```

---

### 6. Skill Gap Analysis
After recommendation, show the user which skills they're missing for each role:

```python
required = set(internship["required_skills"].lower().split(", "))
user     = set(user_skills.lower().split(", "))
missing  = required - user
# → "To qualify for this role, you could learn: Docker, Kubernetes"
```

---

## 📦 Tech Stack Summary

| Layer       | Technology                              |
|-------------|------------------------------------------|
| Backend     | FastAPI, Uvicorn, Pydantic               |
| ML/NLP      | scikit-learn (TF-IDF), numpy, pandas     |
| PDF Parsing | PyMuPDF (fitz), pdfminer.six             |
| Frontend    | Vanilla HTML5, CSS3, JavaScript (ES2022) |
| Storage     | JSON (dataset), pickle (model cache)     |
| Optional    | sentence-transformers, faiss-cpu         |

---

## 🎓 Placement Tips

- Mention **"cosine similarity on TF-IDF vectors"** in interviews — shows NLP fluency
- Describe the **preprocessing pipeline** (stop words, synonym normalisation, n-grams)
- Talk about **why you'd upgrade to embeddings** and what the trade-offs are
- The FAISS section shows awareness of **production-scale search**
- The LLM explanation section shows **GenAI integration awareness**

---

*Built with ❤️ for placement season. Good luck! ◈*
