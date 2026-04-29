"""
recommender.py
--------------
Orchestrates the full recommendation pipeline:
  1. Build / load TF-IDF model from internship corpus
  2. Accept a user profile (skills + interests + resume text)
  3. Return ranked recommendations with explanations
"""

import os
import re
from typing import List, Dict, Optional

from modules.preprocessor import (
    preprocess,
    build_internship_corpus,
    load_internships,
    extract_skills_from_text,
    KNOWN_SKILLS,
)
from modules.feature_extractor import FeatureExtractor


class RecommendationEngine:
    """
    Singleton-friendly recommendation engine.

    Call `initialize()` once at app startup, then call `recommend()` per request.
    """

    def __init__(self):
        self.internships: List[Dict] = []
        self.extractor = FeatureExtractor()
        self._ready = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def initialize(self, force_retrain: bool = False) -> None:
        """
        Load internship data and prepare the TF-IDF model.
        Tries to restore from cache; falls back to training from scratch.
        """
        self.internships = load_internships()

        if not force_retrain and self.extractor.load():
            print("[RecommendationEngine] Loaded TF-IDF model from cache.")
        else:
            print("[RecommendationEngine] Training TF-IDF model ...")
            corpus = build_internship_corpus(self.internships)
            self.extractor.fit(corpus)
            self.extractor.save()
            print("[RecommendationEngine] Model trained and cached.")

        self._ready = True

    # ------------------------------------------------------------------
    # Core recommendation logic
    # ------------------------------------------------------------------

    def recommend(
        self,
        skills: str = "",
        interests: str = "",
        resume_text: str = "",
        location: str = "",
        domain: str = "",
        top_k: int = 8,
    ) -> List[Dict]:
        """
        Generate ranked internship recommendations for a user.

        Parameters
        ----------
        skills        : comma-separated or free-form skills text
        interests     : free-form interests / preferred domain text
        resume_text   : raw extracted text from an uploaded PDF resume
        location      : optional location filter (city name)
        domain        : optional domain filter (e.g. "AI/ML", "Web Development")
        top_k         : number of recommendations to return

        Returns
        -------
        List of dicts containing internship details + relevance_score + explanation
        """
        if not self._ready:
            raise RuntimeError("Engine not initialised. Call initialize() first.")

        # ---- Build user profile string --------------------------------
        user_profile = self._build_user_profile(skills, interests, resume_text)
        if not user_profile.strip():
            return []

        preprocessed_profile = preprocess(user_profile, remove_stops=True)

        # ---- Vectorise and score -------------------------------------
        query_vector = self.extractor.transform_query(preprocessed_profile)
        scores = self.extractor.similarity_scores(query_vector)

        # ---- Apply optional filters ----------------------------------
        candidate_indices = list(range(len(self.internships)))
        if location:
            candidate_indices = [
                i for i in candidate_indices
                if location.lower() in self.internships[i].get("location", "").lower()
                or self.internships[i].get("location", "").lower() == "remote"
            ]
        if domain:
            candidate_indices = [
                i for i in candidate_indices
                if domain.lower() in self.internships[i].get("domain", "").lower()
            ]

        # ---- Rank candidates -----------------------------------------
        ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)
        top_indices = ranked[:top_k]

        # ---- Get key terms for explanation ---------------------------
        key_terms = self.extractor.get_top_feature_terms(query_vector, n=15)

        # ---- Build response ------------------------------------------
        results = []
        max_score = max([scores[i] for i in top_indices]) if top_indices else 1.0
        for idx in top_indices:
            internship = dict(self.internships[idx])
            raw_score = float(scores[idx])

            # Convert to a human-readable percentage (0–100)
            if max_score > 0:
                normalized_score = (raw_score / max_score) * 100
            else:
                normalized_score = 0.0

            internship["relevance_score"] = round(normalized_score, 1)
            internship["match_grade"] = self._grade(raw_score)
            internship["explanation"] = self._explain(
                internship, skills, interests, resume_text, key_terms
            )
            results.append(internship)

        # Ensure at least something is returned even if scores are 0
        # (can happen when profile is very sparse)
        if not results and not location and not domain:
            results = [
                {**dict(self.internships[i]), "relevance_score": 0.0,
                 "match_grade": "Low", "explanation": "Insufficient profile data to score."}
                for i in range(min(top_k, len(self.internships)))
            ]

        return results

    # ------------------------------------------------------------------
    # Profile building
    # ------------------------------------------------------------------

    def _build_user_profile(self, skills: str, interests: str, resume_text: str) -> str:
        """
        Concatenate all user inputs into one weighted profile string.
        Skills are repeated to boost their weight in TF-IDF.
        """
        parts = []

        # Skills (repeated 3× for extra weight)
        if skills.strip():
            parts.append((skills + " ") * 3)

        # Interests / domain
        if interests.strip():
            parts.append(interests)

        # Resume text (single occurrence – provides context)
        if resume_text.strip():
            # Extract recognised skills first and add them with repetition
            extracted = extract_skills_from_text(resume_text)
            if extracted:
                parts.append((" ".join(extracted) + " ") * 2)
            parts.append(resume_text[:3000])  # cap to first 3000 chars

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Explanation generator
    # ------------------------------------------------------------------

    def _explain(
        self,
        internship: Dict,
        skills: str,
        interests: str,
        resume_text: str,
        key_terms: List[str],
    ) -> str:
        """
        Generate a human-readable explanation for why an internship was recommended.
        Matches user skills against the internship's required skills.
        """
        # required_skills may be a list or a string — normalise to lowercase string
        required_raw = internship.get("required_skills", "")
        if isinstance(required_raw, list):
            required_raw = " ".join(required_raw)
        required = required_raw.lower()
        user_text = (skills + " " + interests + " " + resume_text).lower()

        # Find overlapping skills
        matched_skills = []
        for skill in KNOWN_SKILLS:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, required) and re.search(pattern, user_text):
                matched_skills.append(skill.title())

        reasons = []

        if matched_skills:
            skill_list = ", ".join(matched_skills[:5])
            reasons.append(f"Your skills in **{skill_list}** directly match this role's requirements.")

        domain_val = internship.get("domain", "")
        if domain_val and domain_val.lower() in user_text:
            reasons.append(f"Your interest in **{domain_val}** aligns with this internship's focus area.")

        if key_terms:
            relevant_terms = [
                t for t in key_terms
                if t in required and len(t) > 3
            ][:3]
            if relevant_terms:
                reasons.append(
                    f"Key profile terms — *{', '.join(relevant_terms)}* — appear prominently in this job description."
                )

        if not reasons:
            reasons.append("This internship broadly matches your overall profile keywords.")

        return " ".join(reasons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grade(score: float) -> str:
        """Map a cosine similarity score to a friendly grade label."""
        if score >= 0.35:
            return "Excellent"
        elif score >= 0.20:
            return "Strong"
        elif score >= 0.10:
            return "Good"
        elif score >= 0.04:
            return "Fair"
        else:
            return "Low"

    def get_all_domains(self) -> List[str]:
        """Return unique domain values from the dataset."""
        domains = sorted({i.get("domain", "") for i in self.internships if i.get("domain")})
        return domains

    def get_all_locations(self) -> List[str]:
        """Return unique location values from the dataset."""
        locs = sorted({i.get("location", "") for i in self.internships if i.get("location")})
        return locs