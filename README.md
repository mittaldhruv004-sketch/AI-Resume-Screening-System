# HireSense AI — Intelligent Resume Screening & Candidate Analysis

A Streamlit-based decision-support application that compares resumes with a job description, extracts structured evidence, ranks candidate profiles and explains the score components.

> **Responsible-use note:** HireSense AI is a screening aid, not an automated hiring decision-maker. Scores are heuristic indicators, not validated predictions of job performance. Recruiters must review source resumes, use consistent criteria and avoid protected or irrelevant personal characteristics.

## What's upgraded

- **Full-text skill coverage:** canonical skill groups are searched across the complete resume instead of relying on the resume's top-frequency phrases.
- **Safer education detection:** degree detection uses context and word boundaries to avoid false positives from phrases such as `MS Office`, `Scrum Master` and `database`.
- **Experience extraction:** explicit years are detected first; otherwise date ranges are parsed, education ranges are filtered and overlapping employment periods are merged.
- **Explainable scoring:** normalized weighted score, individual component scores, matched skills, missing skills and per-factor contributions.
- **Chunk-aware semantic matching:** long documents are split into manageable chunks before Sentence Transformer embeddings are averaged. If the model package/model is unavailable, a TF-IDF fallback is used.
- **Document safeguards:** 5 MB per file, PDF page cap, text-length cap, basic PDF/DOCX signature checks, DOCX table extraction and scanned-PDF warnings.
- **Duplicate detection:** identical normalized resume text is skipped within an analysis.
- **Safer exports:** CSV formula injection prefixes are neutralized.
- **Recruiter workspace:** ranking table, score analytics, candidate insights, minimum-score filter and CSV export.
- **Demo mode:** three fictional sample candidates and a sample job description are bundled for demonstrations.

## Quick start

Python 3.10–3.12 is recommended.

```bash
# 1. Create an environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

The first semantic-model run may download `sentence-transformers/all-MiniLM-L6-v2`. If it cannot be loaded, the app falls back to local TF-IDF matching. For a CPU-only PyTorch installation, follow the official PyTorch selector at https://pytorch.org/get-started/locally/ before installing sentence-transformers.

## Demo

1. Start the app.
2. Keep **Use included demo candidates** enabled.
3. Use the pre-filled Data Analyst / Junior Data Scientist job description.
4. Click **Analyze candidates**.
5. Explore Candidate ranking, Analytics, Candidate insights and Methodology tabs.

All sample people and contact details are fictional.

## Project structure

```text
AI-Resume-Screening-System/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .streamlit/config.toml
├── sample_data/
│   ├── sample_job_description.txt
│   ├── Aarav_Sharma_Data_Analyst.txt
│   ├── Meera_Iyer_Data_Scientist.txt
│   └── Kabir_Khan_Business_Analyst.txt
├── screenshots/                 # UI preview images
├── tests/
│   └── test_core.py
└── src/
    ├── scoring.py
    ├── processing/
    │   ├── nlp.py
    │   └── embedding.py
    └── utils/
        ├── parser.py
        ├── resume_extractor.py
        ├── keyword_mapping.py
        └── security.py
```

## Scoring model

The default profile uses these configurable weights:

| Component | Default weight |
|---|---:|
| Semantic similarity | 0.35 |
| Required skill coverage | 0.35 |
| Experience match | 0.15 |
| Education match | 0.10 |
| Certification match | 0.05 |

The overall score is a normalized weighted average:

\[S = \frac{\sum_i w_i s_i}{\sum_i w_i}\]

If a job description does not specify an experience, education or certification requirement, the corresponding component uses a neutral score. These are transparent product heuristics and should be calibrated and validated on a representative, consented dataset before real recruitment use.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover education false positives, education detection, newline preservation, full-text skill extraction, keyword coverage, explicit experience extraction and zero-weight validation.

## Privacy and deployment

- Uploaded documents are processed in application memory; this project does not intentionally persist uploaded resume files.
- Do not deploy publicly with real candidate data without authentication, access controls, secure hosting, retention policy, consent and legal/privacy review.
- The included demo data is fictional. Remove real personal data from logs and exports before sharing.
- The parser enforces file-size, PDF-page and extracted-text limits. Scanned PDFs need OCR before upload.

## License and attribution

The original repository's MIT license is retained. Review the `LICENSE` file and preserve the original copyright and permission notice when redistributing modified versions.
