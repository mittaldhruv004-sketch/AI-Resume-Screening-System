import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

import io
import hashlib
from pathlib import Path
from typing import Dict, List
import pandas as pd
import streamlit as st
import plotly.express as px

from src.utils.parser import extract_text_from_file
from src.utils.resume_extractor import extract_all_resume_features
from src.processing.nlp import extract_keywords, extract_skills, normalize_text
from src.processing.embedding import get_embedder, embed_texts, cosine_similarity_matrix
from src.scoring import compute_scores
from src.utils.security import csv_safe

st.set_page_config(page_title="HireSense AI | Resume Intelligence", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

DEFAULT_WEIGHTS = {"similarity": 0.35, "keywords": 0.35, "experience": 0.15, "education": 0.10, "certs": 0.05}
PRESETS = {
    "Balanced": DEFAULT_WEIGHTS,
    "Senior role": {"similarity": 0.25, "keywords": 0.25, "experience": 0.35, "education": 0.10, "certs": 0.05},
    "Entry level": {"similarity": 0.35, "keywords": 0.40, "experience": 0.05, "education": 0.15, "certs": 0.05},
    "Cloud role": {"similarity": 0.25, "keywords": 0.30, "experience": 0.15, "education": 0.05, "certs": 0.25},
}
SAMPLE_DIR = Path(__file__).parent / "sample_data"

@st.cache_resource(show_spinner=False)
def load_embedder():
    return get_embedder()


def set_preset(name: str):
    values = PRESETS[name]
    for key, value in [("similarity", values["similarity"]), ("keywords", values["keywords"]), ("experience", values["experience"]), ("education", values["education"]), ("certs", values["certs"])]:
        st.session_state[f"weight_{key}"] = value
    st.session_state["active_preset"] = name


def sample_records():
    records = []
    for path in sorted(SAMPLE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        records.append((path.name, text))
    return records


def safe_csv(df: pd.DataFrame) -> bytes:
    export = df.copy()
    for col in export.columns:
        if export[col].dtype == object:
            export[col] = export[col].map(csv_safe)
    return export.to_csv(index=False).encode("utf-8-sig")


def analyze(jd_text: str, raw_records: List[tuple], weights: Dict[str, float], top_k: int):
    if len(jd_text) > 100_000:
        raise ValueError("Job description exceeds the 100,000-character limit.")
    if len(raw_records) > 50:
        raise ValueError("Please analyze no more than 50 resumes at a time.")
    jd_clean = normalize_text(jd_text)
    jd_keywords = extract_keywords(jd_clean, top_k=top_k)
    records, seen_hashes = [], set()
    skipped = []
    name_counts = {}
    for filename, raw_text in raw_records:
        name_counts[filename] = name_counts.get(filename, 0) + 1
        candidate_name = filename if name_counts[filename] == 1 else f"{filename} ({name_counts[filename]})"
        text = normalize_text(raw_text, preserve_newlines=True)
        digest = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            skipped.append((filename, "Duplicate resume content")); continue
        seen_hashes.add(digest)
        if len(text) < 80:
            skipped.append((filename, "Very little readable text")); continue
        features = extract_all_resume_features(text)
        records.append({"filename": candidate_name, "text": text, "keywords": extract_skills(text),
                        "experience_years": features["experience_years"], "education": features["education"],
                        "certifications": features["certifications"], "recent_roles": features["recent_roles"]})
    if not records:
        raise ValueError("No usable resumes were found. Check the file contents and formats.")
    embedder = load_embedder()
    embeddings = embed_texts(embedder, [jd_clean] + [r["text"] for r in records])
    similarity = cosine_similarity_matrix(embeddings)
    result = compute_scores(jd_clean, jd_keywords, records, similarity,
        weights["similarity"], weights["keywords"], weights["experience"], weights["education"], weights["certs"])
    return result, jd_keywords, records, skipped, ("Sentence Transformers" if embedder != "tfidf" else "TF-IDF offline fallback")


def main():
    st.markdown("""<style>
    .block-container{padding-top:1.6rem;padding-bottom:2rem;max-width:1440px}
    [data-testid="stMetric"]{background:linear-gradient(135deg,rgba(40,80,140,.13),rgba(30,160,150,.08));padding:16px;border:1px solid rgba(128,150,180,.18);border-radius:14px}
    .hero{padding:22px 24px;border-radius:18px;background:linear-gradient(115deg,#10233f,#174e68 62%,#167c79);color:white;margin-bottom:20px}
    .hero h1{font-size:2.15rem;margin:0 0 5px 0}.hero p{color:#d7e9f1;margin:0}
    .small-muted{color:#8291a7;font-size:.85rem}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🧭 HireSense AI</h1><p>Resume intelligence · Explainable candidate matching · Recruiter workspace</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Evaluation settings")
        st.caption("Choose how each evidence signal contributes to the overall match.")
        cols = st.columns(2)
        for i, name in enumerate(PRESETS):
            if cols[i % 2].button(name, key=f"preset_{name}", use_container_width=True, on_click=set_preset, args=(name,)):
                pass
        for key, label in [("similarity", "Semantic similarity"), ("keywords", "Required skills"), ("experience", "Experience"), ("education", "Education"), ("certs", "Certifications")]:
            st.slider(label, 0.0, 1.0, DEFAULT_WEIGHTS[key], 0.05, key=f"weight_{key}")
        weights = {key: float(st.session_state.get(f"weight_{key}", value)) for key, value in DEFAULT_WEIGHTS.items()}
        total_weight = sum(weights.values())
        st.caption(f"Active weight total: {total_weight:.2f}")
        st.divider()
        top_k = st.slider("Skill categories to consider", 5, 30, 20)
        st.caption("Files are processed in memory for this session. Avoid uploading sensitive candidate data to an untrusted deployment.")

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("### 1. Job description")
        default_jd = (SAMPLE_DIR / "sample_job_description.txt").read_text(encoding="utf-8") if (SAMPLE_DIR / "sample_job_description.txt").exists() else ""
        jd_text = st.text_area("Paste the role requirements", value=default_jd, height=220, label_visibility="collapsed", placeholder="Paste responsibilities, required skills and experience...")
    with right:
        st.markdown("### 2. Candidate resumes")
        uploaded = st.file_uploader("Add PDF, DOCX or TXT resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, help="Maximum 5 MB per file; PDF up to 25 pages.")
        use_demo = st.toggle("Use included demo candidates", value=True, help="If you upload files, the uploaded resumes take priority.")
        if use_demo:
            st.info("Demo mode loads three fictional sample profiles bundled with the project.")
        if uploaded:
            st.caption(f"{len(uploaded)} file(s) selected")
        st.markdown("<div class='small-muted'>Supported: PDF · DOCX · TXT</div>", unsafe_allow_html=True)

    run_col, info_col = st.columns([1, 3])
    with run_col:
        run = st.button("✨ Analyze candidates", type="primary", use_container_width=True, disabled=(total_weight <= 0 or not jd_text.strip()))
    with info_col:
        st.caption("Scores are decision-support signals, not hiring decisions. Review original resumes and apply consistent human judgement.")

    if run:
        raw_records = []
        if use_demo and not uploaded:
            raw_records = sample_records()
        else:
            for item in uploaded or []:
                try:
                    raw_records.append((item.name, extract_text_from_file(item.name, io.BytesIO(item.getvalue()))))
                except Exception:
                    st.warning("One selected document could not be processed. Check its format, readability and upload limits.")
        if not raw_records:
            st.error("Upload at least one resume or enable demo candidates.")
            return
        with st.spinner("Extracting evidence and comparing candidate profiles..."):
            try:
                result, jd_keywords, records, skipped, model_name = analyze(jd_text, raw_records, weights, top_k)
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")
                return
        st.session_state["analysis_result"] = result
        st.session_state["analysis_jd_keywords"] = jd_keywords
        st.session_state["analysis_skipped"] = skipped
        st.session_state["analysis_model"] = model_name
        st.session_state["analysis_jd"] = jd_text

    if "analysis_result" not in st.session_state:
        st.markdown("---")
        st.markdown("### Your workflow")
        a, b, c = st.columns(3)
        a.markdown("**01 · Parse**\n\nExtract readable text and structured career evidence from each resume.")
        b.markdown("**02 · Match**\n\nCompare skills, context, experience and qualifications against the role.")
        c.markdown("**03 · Explain**\n\nInspect match evidence, missing skills and score contributions before review.")
        return

    result = st.session_state["analysis_result"]
    st.divider()
    st.markdown("### Screening overview")
    best = result.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates analyzed", len(result))
    m2.metric("Top match", f"{best.total_score:.0%}")
    m3.metric("Average match", f"{result.total_score.mean():.0%}")
    m4.metric("Matching engine", st.session_state.get("analysis_model", "N/A"))

    tab_rank, tab_compare, tab_details, tab_method = st.tabs(["🏆 Candidate ranking", "📊 Analytics", "🔎 Candidate insights", "🧠 Methodology"])
    with tab_rank:
        st.markdown("#### Ranked candidate shortlist")
        threshold = st.slider("Minimum match score", 0, 100, 0, 5, key="min_score_filter")
        filtered = result[result.total_score >= threshold / 100]
        st.dataframe(filtered[["rank", "filename", "total_score", "similarity", "keyword_coverage", "experience_score", "education_score", "certification_score", "experience_years", "education", "recent_roles", "matched_keywords", "missing_keywords"]], use_container_width=True, hide_index=True,
            column_config={"total_score": st.column_config.ProgressColumn("Overall match", min_value=0, max_value=1, format="%.0%%"), "similarity": st.column_config.NumberColumn("Semantic", format="%.0%%"), "keyword_coverage": st.column_config.NumberColumn("Skills", format="%.0%%"), "experience_score": st.column_config.NumberColumn("Experience", format="%.0%%"), "education_score": st.column_config.NumberColumn("Education", format="%.0%%"), "certification_score": st.column_config.NumberColumn("Certifications", format="%.0%%")})
        st.download_button("⬇️ Download shortlist (CSV)", safe_csv(result), "hiresense_candidate_scores.csv", "text/csv")
        if st.session_state.get("analysis_skipped"):
            with st.expander("Files not included"):
                st.write(st.session_state["analysis_skipped"])
    with tab_compare:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Overall candidate match")
            fig = px.bar(result, x="total_score", y="filename", orientation="h", color="total_score", range_x=[0, 1], text=result.total_score.map(lambda x: f"{x:.0%}"), color_continuous_scale="Teal")
            fig.update_layout(height=360, yaxis={"categoryorder": "total ascending"}, xaxis_title="Weighted match score", yaxis_title="", coloraxis_showscale=False, margin=dict(l=8,r=8,t=12,b=8))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Evidence score breakdown")
            melted = result.melt(id_vars="filename", value_vars=["similarity", "keyword_coverage", "experience_score", "education_score", "certification_score"], var_name="factor", value_name="score")
            fig2 = px.bar(melted, x="filename", y="score", color="factor", barmode="group", range_y=[0, 1])
            fig2.update_layout(height=360, xaxis_title="", yaxis_title="Factor score", legend_title="")
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("#### Skill coverage across the shortlist")
        all_missing = []
        for _, row in result.iterrows():
            for skill in str(row.missing_keywords).split(", "):
                if skill and skill != "None": all_missing.append({"Candidate": row.filename, "Missing skill": skill})
        if all_missing:
            st.dataframe(pd.DataFrame(all_missing), use_container_width=True, hide_index=True)
        else:
            st.success("All detected required skill categories appear in the candidate profiles.")
    with tab_details:
        selected = st.selectbox("Select a candidate", result.filename.tolist())
        row = result[result.filename == selected].iloc[0]
        st.markdown(f"#### #{int(row['rank'])} · {selected}")
        x1, x2, x3 = st.columns(3)
        x1.metric("Overall match", f"{row.total_score:.0%}")
        x2.metric("Experience detected", f"{row.experience_years:g} yrs" if pd.notna(row.experience_years) else "Not detected")
        x3.metric("Education", row.education)
        matched_col, missing_col = st.columns(2)
        with matched_col:
            st.markdown("**Matched skill categories**")
            st.success(row.matched_keywords)
        with missing_col:
            st.markdown("**Potential skill gaps**")
            st.warning(row.missing_keywords)
        st.markdown("**Recent role titles detected**")
        st.write(row.recent_roles)
        st.markdown("**Score contribution breakdown**")
        contrib = pd.DataFrame({"Factor": ["Semantic", "Skills", "Experience", "Education", "Certifications"], "Contribution": [row.similarity * weights["similarity"] / sum(weights.values()), row.keyword_coverage * weights["keywords"] / sum(weights.values()), row.experience_score * weights["experience"] / sum(weights.values()), row.education_score * weights["education"] / sum(weights.values()), row.certification_score * weights["certs"] / sum(weights.values())]})
        st.bar_chart(contrib.set_index("Factor"))
        st.caption("Detected certifications: " + str(row.certifications))
    with tab_method:
        st.markdown("#### How HireSense AI evaluates a resume")
        st.write("The score is a normalized weighted average of five transparent components. Skill coverage is computed from the full extracted resume text, not only a short list of frequent phrases.")
        st.latex(r"S = \frac{\sum_i w_i s_i}{\sum_i w_i}")
        st.markdown("- **Semantic similarity:** contextual similarity between job description and resume. Long documents are embedded in overlapping-sized chunks before averaging.")
        st.markdown("- **Required skills:** coverage of canonical skill categories found in the complete text.")
        st.markdown("- **Experience:** compares an explicit minimum requirement with detected years; overlapping date ranges are merged.")
        st.markdown("- **Education:** boundary-aware degree detection avoids false matches such as `MS Office` or `database`.")
        st.markdown("- **Certifications:** checks recognized certification names only when the job description asks for them.")
        st.info("A high score is not a prediction of job performance. Human review, structured interviews and validation are required. Do not use protected or irrelevant personal characteristics in hiring decisions.")

if __name__ == "__main__":
    main()
