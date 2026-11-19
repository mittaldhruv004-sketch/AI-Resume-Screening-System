# Changelog

## HireSense AI upgrade

- Rebranded the interface as HireSense AI while retaining the original Streamlit/Python foundation.
- Replaced fragile degree substring matching with contextual degree detection.
- Reworked experience extraction to merge overlapping date intervals and filter education dates.
- Preserved resume line structure for role extraction.
- Added full-document skill-group detection and missing-skill reporting.
- Added chunk-aware semantic embeddings and TF-IDF fallback.
- Rebuilt scoring as normalized weighted components with contribution explanations.
- Added upload validation, DOCX table extraction, duplicate detection and CSV formula sanitization.
- Fixed preset slider state using keyed session-state values.
- Added sample candidates, demo job description, tests and updated documentation.

## Resume matching regression fix (2026-11-19)
- Expanded the canonical skill taxonomy to recognize C, C++, Flask, HTML, CSS, bcrypt, REST APIs, Operating Systems and Computer Networks.
- Removed the remaining top-k truncation from canonical job-description skill extraction; all recognized skill categories in the full text are retained.
- Added regression tests for the supplied resume's education and technical-skill patterns and for JD/resume alias matching.
- Verified education extraction for M.Tech, M.Sc and B.Sc; no change to the weighted scoring formula or score weights.
