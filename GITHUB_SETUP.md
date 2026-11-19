# GitHub setup

1. Create a new repository for your upgraded HireSense AI project.
2. Review the MIT license and retain the original copyright and permission notice.
3. Confirm `.env` and `.streamlit/secrets.toml` are excluded by `.gitignore`.
4. Do not commit resumes, candidate exports, credentials, private datasets or local machine paths.
5. Add source code, fictional sample data, tests and documentation.
6. Run `python -m pytest -q` before pushing.

Suggested commit sequence:

- `fix: improve resume extraction and scoring accuracy`
- `feat: add explainable candidate insights`
- `feat: add recruiter analytics and demo dataset`
- `docs: document setup, evaluation and responsible use`
