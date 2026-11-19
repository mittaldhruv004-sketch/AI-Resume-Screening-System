import numpy as np
from src.utils.resume_extractor import extract_education, extract_experience_years
from src.processing.nlp import normalize_text, extract_skills
from src.scoring import _education_match, _keyword_coverage, compute_scores


def test_education_does_not_match_common_words():
    assert extract_education("I use MS Office and work with databases because I am a Scrum Master") == {"phd": False, "masters": False, "bachelors": False, "diploma": False}


def test_education_detects_degree():
    assert extract_education("Bachelor of Technology in Computer Science") ["bachelors"]


def test_education_jd_uses_boundaries():
    assert _education_match("Experience with database systems is required", {}) == 0.5


def test_newlines_can_be_preserved():
    assert normalize_text("Role\nData Analyst", preserve_newlines=True) == "Role\nData Analyst"


def test_skill_extraction_uses_whole_text():
    assert "python" in extract_skills("A long profile. " + "general text " * 100 + "Python, SQL and AWS")


def test_keyword_coverage_uses_full_skill_set():
    assert _keyword_coverage(["python", "sql"], ["python", "sql", "aws"]) == 1.0


def test_experience_direct_mention():
    assert extract_experience_years("More than 5 years of professional experience") == 5.0


def test_zero_weights_rejected():
    try:
        compute_scores("", [], [], np.zeros((1, 1)), 0, 0)
    except ValueError:
        return
    assert False, "Expected zero-weight validation"


def test_ambiguous_degree_abbreviations_are_not_false_positives():
    result = extract_education("Business Analyst (BA); it will be reviewed; MS Office; Scrum Master")
    assert not result["bachelors"]
    assert not result["masters"]


def test_date_ranges_merge_overlapping_periods():
    # Overlap must not be counted twice; a degree period must not count as employment.
    years = extract_experience_years("Education: 2017 - 2021 B.Tech\nExperience: Jan 2020 - Dec 2022\nExperience: Jan 2022 - Dec 2023")
    assert years is not None
    assert 3.0 <= years <= 4.0


def test_certification_extraction_handles_vendor_prefix():
    from src.utils.resume_extractor import extract_certifications
    assert extract_certifications("Microsoft Certified: Power BI Data Analyst Associate")


def test_csv_formula_prefix_is_neutralized():
    from src.utils.security import csv_safe
    assert csv_safe("=HYPERLINK('x')").startswith("'")


def test_scrum_master_is_not_a_degree_requirement():
    assert _education_match("The role works with an Agile Scrum Master", {}) == 0.5


def test_docx_parser_reads_tables():
    from io import BytesIO
    from docx import Document
    from src.utils.parser import extract_text_from_file
    doc = Document()
    doc.add_paragraph("Candidate profile")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Python"
    table.cell(0, 1).text = "SQL"
    buf = BytesIO()
    doc.save(buf)
    text = extract_text_from_file("candidate.docx", BytesIO(buf.getvalue()))
    assert "Python" in text and "SQL" in text


def test_unambiguous_be_degree_format_is_detected():
    assert extract_education("B.E. in Mechanical Engineering") ["bachelors"]
    assert extract_education("it will be useful") ["bachelors"] is False


def test_resume_specific_skill_and_education_regression():
    from src.utils.parser import extract_text_from_file
    from src.utils.resume_extractor import extract_all_resume_features
    from io import BytesIO
    resume = (
        "M.Tech Data Analytics\nM.Sc Computer Science\nB.Sc Computer Science\n"
        "Programming Languages: C, C++\nDatabases: MySQL, SQLite\n"
        "Web Technologies: HTML, CSS, Flask\n"
        "Libraries: Scikit-learn, Pandas, NumPy\n"
        "Developed a web-based system using Python Flask and MySQL."
    )
    skills = set(extract_skills(resume))
    assert {"c", "cpp", "python", "flask", "html", "css", "sql", "pandas", "numpy", "scikit_learn"}.issubset(skills)
    education = extract_all_resume_features(resume)["education"]
    assert education["masters"] and education["bachelors"]


def test_jd_and_resume_skill_aliases_match_canonically():
    jd = "Required: Python, Flask, MySQL, HTML, CSS, C++ and Pandas."
    resume = "Built a Flask application with Python, MySQL, HTML, CSS, C++ and Pandas."
    assert _keyword_coverage(extract_skills(jd), extract_skills(resume)) == 1.0
