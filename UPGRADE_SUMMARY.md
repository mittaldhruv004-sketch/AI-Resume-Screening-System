# HireSense AI upgrade summary

## Functional fixes

- Education matching now avoids common-word and job-title false positives (`be`, `MS Office`, `Scrum Master`, `BA`, `database`).
- Work-history date ranges are captured directly, filtered against education context and merged to avoid double-counting overlapping jobs.
- Newlines are preserved for role extraction.
- Skill matching checks the complete resume and uses a more granular skill taxonomy (Python, Pandas, NumPy, scikit-learn and related tools are separate categories).
- Certification detection recognizes bounded credential patterns and is neutral when the job description has no certification requirement.
- Experience stays numeric in the result data.
- Scoring rejects zero total weight and normalizes active weights.
- Candidate names are disambiguated when uploaded filenames collide.

## Product additions

- HireSense AI branding and recruiter-focused interface.
- Four scoring presets and configurable weights.
- Candidate ranking and minimum-score filter.
- Missing-skill report and recent role display.
- Per-candidate score contribution breakdown.
- Recruitment analytics and CSV export.
- Three fictional demo resumes and a sample job description.
- Duplicate resume content detection.

## Security and reliability

- File-size, page-count, character-count and batch-size limits.
- PDF/DOCX signature checks and DOCX archive expansion cap.
- DOCX table extraction.
- Scanned PDF warning when no text can be extracted.
- CSV formula-injection mitigation.
- Removed runtime spaCy model downloads and deprecated PyPDF2 dependency.
- Added offline TF-IDF fallback for semantic matching.
- Added automated tests.

## Validation performed in the build environment

- Python source compilation: passed.
- Unit tests: 14 passed.
- Demo-data extraction and scoring pipeline: executed successfully with the TF-IDF fallback.
- Streamlit browser runtime: not executed in this build environment because Streamlit was not installed and package installation was blocked by unavailable outbound network access. The PNGs in `screenshots/` are clearly labeled static UI previews, not live Streamlit captures.
