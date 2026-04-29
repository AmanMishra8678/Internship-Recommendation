"""
preprocessor.py
---------------
Text cleaning and normalization module.
Handles all NLP preprocessing: lowercasing, stop word removal,
lemmatization, and skill extraction from raw text.
"""

import re
import string
import json
import os
from typing import List, Dict

# ---------------------------------------------------------------------------
# Lightweight stop-word list (avoids requiring NLTK data downloads)
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "it", "they", "my", "our", "your",
    "his", "her", "its", "their", "me", "us", "him", "them", "as", "if",
    "so", "up", "out", "not", "no", "also", "into", "about", "than",
    "then", "just", "more", "some", "any", "all", "each", "both", "very",
    "well", "work", "using", "use", "used", "build", "building", "built",
    "develop", "developing", "developed", "implement", "implementing",
    "design", "designing", "create", "creating", "write", "writing"
}

# Common tech skill synonyms → canonical form
SKILL_SYNONYMS: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "db": "database",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node.js",
    "nodejs": "node.js",
    "next.js": "next.js",
    "nextjs": "next.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "nosql": "nosql",
    "aws": "aws",
    "gcp": "google cloud",
    "azure": "azure",
    "ci/cd": "ci/cd",
    "oop": "object oriented programming",
    "dsa": "data structures and algorithms",
    "algo": "algorithms",
    "stat": "statistics",
    "stats": "statistics",
    "viz": "data visualization",
}


def clean_text(text) -> str:
    """
    Clean and normalise raw text.
    Now fully safe against list / None / any input.
    """

    # 🔥 HARD GUARD (this will stop your crash 100%)
    if isinstance(text, list):
        text = " ".join(map(str, text))
    elif text is None:
        text = ""
    else:
        text = str(text)

    text = text.lower()

    import re
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove special chars
    text = re.sub(r"[^\w\s\-\./]", " ", text)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """Remove common stop words from cleaned text."""
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(filtered)


def normalize_skills(text: str) -> str:
    """
    Replace known skill synonyms with their canonical forms
    so that 'js' and 'javascript' are treated identically.
    """
    tokens = text.split()
    normalized = [SKILL_SYNONYMS.get(t, t) for t in tokens]
    return " ".join(normalized)


def preprocess(text, remove_stops: bool = True) -> str:
    """
    Full preprocessing pipeline.
    Now fully safe against list / None inputs.
    """

    # 🔥 HARD FIX (this is what you're missing)
    if isinstance(text, list):
        text = " ".join(map(str, text))
    elif text is None:
        text = ""
    else:
        text = str(text)

    text = clean_text(text)

    if remove_stops:
        text = remove_stopwords(text)

    text = normalize_skills(text)

    return text


# ---------------------------------------------------------------------------
# Skill extraction heuristics
# ---------------------------------------------------------------------------

# Hand-curated master skill list used for fuzzy matching against resume text
KNOWN_SKILLS: List[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "kotlin", "swift", "scala", "ruby", "php", "r", "matlab",
    "bash", "shell", "perl", "dart", "haskell",
    # Web
    "react", "angular", "vue", "next.js", "node.js", "express", "django",
    "flask", "fastapi", "spring boot", "laravel", "html", "css", "sass",
    "tailwind", "graphql", "rest api", "websocket",
    # Data / ML / AI
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "scikit-learn", "pytorch",
    "tensorflow", "keras", "hugging face", "transformers", "bert", "gpt",
    "yolo", "opencv", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "xgboost", "lightgbm", "catboost", "statistics", "linear algebra",
    "time series", "data analysis", "data visualization", "feature engineering",
    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "elasticsearch",
    "cassandra", "firebase", "bigquery", "snowflake",
    # Cloud / DevOps
    "aws", "azure", "google cloud", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "github actions", "ci/cd", "linux", "nginx",
    "apache kafka", "kafka", "airflow", "spark",
    # Mobile
    "android", "ios", "react native", "flutter", "swiftui", "jetpack compose",
    # Security
    "cybersecurity", "ethical hacking", "penetration testing", "burp suite",
    "networking", "cryptography", "owasp",
    # Blockchain
    "solidity", "ethereum", "web3", "smart contracts", "defi",
    # Tools
    "git", "github", "jira", "figma", "adobe xd", "postman", "excel",
    "tableau", "power bi", "latex",
    # Concepts
    "data structures", "algorithms", "system design", "object oriented programming",
    "microservices", "agile", "scrum", "design patterns",
]


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract recognised skill tokens from free-form text (e.g. a resume).
    Returns a deduplicated, sorted list of matched skills.
    """
    text_lower = text.lower()
    found: List[str] = []
    for skill in KNOWN_SKILLS:
        # Use word-boundary matching so "go" doesn't match "algorithm"
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_internships(filepath: str = None) -> List[Dict]:
    """Load internship records from JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "internships.json")
    filepath = os.path.abspath(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_internship_corpus(internships: List[Dict]) -> List[str]:
    """
    Convert a list of internship dictionaries into a list of 
    preprocessed strings for TF-IDF vectorization.
    """
    corpus = []
    for item in internships:
        # FIX: Convert the list of skills into a single string
        skills_raw = item.get("required_skills", [])
        if isinstance(skills_raw, list):
            skills_str = " ".join(skills_raw)
        else:
            skills_str = str(skills_raw)
        
        # FIX: Convert the list of tags into a single string
        tags_raw = item.get("tags", [])
        if isinstance(tags_raw, list):
            tags_str = " ".join(tags_raw)
        else:
            tags_str = str(tags_raw)

        # Now all items are strings, so this .join will NOT crash
        combined = " ".join([
            str(item.get("title", "")),
            str(item.get("company", "")),
            skills_str, skills_str, skills_str,  # Triple weight for skills
            str(item.get("description", "")),
            tags_str     
        ])
        
        corpus.append(preprocess(combined))
        
    return corpus