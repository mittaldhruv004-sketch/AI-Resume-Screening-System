from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.processing.nlp import extract_skills
from src.scoring import compute_scores
from src.utils.resume_extractor import extract_all_resume_features


def test_bundled_demo_pipeline_produces_ranked_candidates():
    sample_dir = Path(__file__).parents[1] / "sample_data"
    jd = (sample_dir / "sample_job_description.txt").read_text(encoding="utf-8")
    resumes = []
    for path in sorted(sample_dir.glob("*.txt")):
        if path.name == "sample_job_description.txt":
            continue
        text = path.read_text(encoding="utf-8")
        resumes.append({"filename": path.name, "text": text, "keywords": extract_skills(text), **extract_all_resume_features(text)})
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([jd] + [r["text"] for r in resumes]).toarray()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T
    result = compute_scores(jd, extract_skills(jd), resumes, similarities, .35, .35, .15, .10, .05)
    assert len(result) == 3
    assert result["rank"].tolist() == [1, 2, 3]
    assert result["total_score"].between(0, 1).all()
    assert "missing_keywords" in result.columns
