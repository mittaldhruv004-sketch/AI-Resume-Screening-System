# Deployment guide

## Local deployment (recommended for project demonstration)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Create a new app in Streamlit Community Cloud.
3. Select the repository, branch and `app.py` as the entry point.
4. Deploy and wait for dependency installation and first model initialization.

For public deployment, do not process real candidate resumes without authentication, access controls, a privacy notice, retention controls and legal review. Public unauthenticated deployments should use fictional demo data only.

## Runtime notes

- The app uses Sentence Transformers when the model can be loaded. First use may require downloading model weights.
- A TF-IDF fallback supports offline operation when Sentence Transformers cannot be loaded.
- Upload limits are configured to 5 MB per file. The parser enforces a 25-page PDF cap and a 100,000-character extracted-text cap.
- Scanned PDFs require OCR before upload.
