"""Explainable, normalized candidate scoring (decision support only)."""
import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _keyword_coverage(jd_keywords: List[str], resume_keywords: List[str]) -> float:
    required, present = set(jd_keywords or []), set(resume_keywords or [])
    return len(required & present) / len(required) if required else 0.5


def _experience_match(jd_text: str, resume_exp_years: Optional[float]) -> float:
    if resume_exp_years is None:
        return 0.5
    patterns = [r"(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?(?:relevant\s*)?(?:professional\s*)?(?:experience|exp)", r"(?:minimum|min|at least|required)\s+(\d+(?:\.\d+)?)\+?\s*years?"]
    required = None
    for pattern in patterns:
        match = re.search(pattern, jd_text or "", re.I)
        if match:
            required = float(match.group(1)); break
    if required is None or required <= 0:
        return 0.5
    return min(1.0, max(0.0, float(resume_exp_years) / required))


def _education_match(jd_text: str, resume_education: Dict[str, bool]) -> float:
    text = jd_text or ""
    requirements = {
        "phd": bool(re.search(r"\b(ph\.?d\.?|doctorate|doctoral)\b", text, re.I)),
        "masters": bool(re.search(r"\b(master(?:'s)?\s+(?:degree|in|of)|postgraduate|m\.?tech|mba|mca)\b", text, re.I)),
        "bachelors": bool(re.search(r"\b(bachelor'?s?|undergraduate|b\.?tech|b\.?sc)\b", text, re.I)),
        "diploma": bool(re.search(r"\bdiploma\b", text, re.I)),
    }
    if not requirements["bachelors"] and re.search(r"(?<![A-Za-z])B\.?E\.?(?![A-Za-z])(?=\s+(?:degree|in|computer|mechanical|electrical|civil))", text):
        requirements["bachelors"] = True
    if not any(requirements.values()):
        return 0.5
    levels = ["diploma", "bachelors", "masters", "phd"]
    required_level = max((levels.index(k) for k, v in requirements.items() if v), default=0)
    achieved = max((levels.index(k) for k, v in (resume_education or {}).items() if v and k in levels), default=-1)
    if achieved < 0:
        return 0.0
    return 1.0 if achieved >= required_level else max(0.0, (achieved + 1) / (required_level + 1))


def _certification_match(jd_text: str, resume_certs: List[str]) -> float:
    jd = (jd_text or "").lower()
    requirement_terms = ["certification", "certified", "certificate", "pmp", "cissp", "ccna", "cka", "aws certified", "azure certification", "google cloud certified"]
    if not any(term in jd for term in requirement_terms):
        return 0.5
    if not resume_certs:
        return 0.0
    meaningful = {w for w in re.findall(r"[a-z0-9+#]+", jd) if len(w) > 2 and w not in {"the", "and", "for", "with", "certified", "certification", "certificate"}}
    for cert in resume_certs:
        cert_words = {w for w in re.findall(r"[a-z0-9+#]+", cert.lower()) if len(w) > 2}
        if cert_words and len(cert_words & meaningful) >= 1:
            return 1.0
    return 0.25


def compute_scores(jd_text: str, jd_keywords: List[str], resumes: List[Dict], sim_matrix: np.ndarray,
                   weight_similarity: float, weight_keywords: float, weight_experience: float = 0.0,
                   weight_education: float = 0.0, weight_certifications: float = 0.0) -> pd.DataFrame:
    weights = {"similarity": max(0.0, weight_similarity), "keyword_coverage": max(0.0, weight_keywords),
               "experience_score": max(0.0, weight_experience), "education_score": max(0.0, weight_education),
               "certification_score": max(0.0, weight_certifications)}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("At least one scoring weight must be greater than zero.")
    rows = []
    sims = sim_matrix[0, 1:] if sim_matrix.ndim == 2 else np.zeros(len(resumes))
    for idx, resume in enumerate(resumes):
        similarity = float(np.clip(sims[idx] if idx < len(sims) else 0.0, 0.0, 1.0))
        present = set(resume.get("keywords", [])); required = set(jd_keywords or [])
        matched, missing = sorted(required & present), sorted(required - present)
        components = {
            "similarity": similarity,
            "keyword_coverage": _keyword_coverage(jd_keywords, resume.get("keywords", [])),
            "experience_score": _experience_match(jd_text, resume.get("experience_years")),
            "education_score": _education_match(jd_text, resume.get("education", {})),
            "certification_score": _certification_match(jd_text, resume.get("certifications", [])),
        }
        total = sum(weights[k] * components[k] for k in weights) / total_weight
        contribution = {f"{k}_contribution": round(weights[k] * components[k] / total_weight, 4) for k in weights}
        edu = resume.get("education", {})
        edu_labels = [name for key, name in [("phd", "PhD"), ("masters", "Masters"), ("bachelors", "Bachelors"), ("diploma", "Diploma")] if edu.get(key)]
        rows.append({"filename": resume.get("filename", "candidate"), **components, "total_score": total,
                     "experience_years": resume.get("experience_years"), "education": ", ".join(edu_labels) or "Not specified",
                     "certifications": ", ".join(resume.get("certifications", [])) or "None detected",
                     "recent_roles": ", ".join(resume.get("recent_roles", [])) or "Not detected",
                     "matched_keywords": ", ".join(matched) or "None", "missing_keywords": ", ".join(missing) or "None",
                     **contribution})
    df = pd.DataFrame(rows).sort_values(["total_score", "similarity"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
