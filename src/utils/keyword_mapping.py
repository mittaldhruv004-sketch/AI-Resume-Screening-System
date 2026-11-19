"""Canonical skill taxonomy and degree vocabulary used by HireSense AI."""
from typing import Dict, List, Set

# Keep distinct skills distinct: Pandas, NumPy and Python are related, not synonyms.
SKILL_GROUPS: Dict[str, List[str]] = {
    "python": ["python", "python3", "python 3"],
    "c": ["c", "c language"],
    "cpp": ["c++", "cpp", "cplusplus"],
    "flask": ["flask"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "operating_systems": ["operating systems", "operating system", "os"],
    "computer_networks": ["computer networks", "computer network", "networking"],
    "bcrypt": ["bcrypt"],
    "rest_api": ["rest api", "restful api", "rest apis", "restful apis"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit_learn": ["scikit-learn", "scikit learn", "sklearn"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "keras"],
    "data_science": ["data science", "machine learning", "ml", "deep learning", "ai", "artificial intelligence", "data analysis", "data analytics"],
    "statistics": ["statistics", "statistical analysis", "statistical testing", "probability"],
    "data_visualization": ["data visualization", "data visualisation", "visualization", "visualisation", "matplotlib", "seaborn"],
    "power_bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "microsoft excel", "spreadsheets"],
    "javascript": ["javascript", "js", "node.js", "nodejs", "react", "vue", "angular", "express", "next.js", "typescript"],
    "java": ["java", "spring", "spring boot", "hibernate", "maven", "gradle"],
    "dotnet": [".net", "dotnet", "c#", "asp.net", "aspnet", "entity framework"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation", "cloudwatch"],
    "azure": ["azure", "microsoft azure", "azure devops", "azure functions"],
    "gcp": ["gcp", "google cloud", "google cloud platform", "gce", "cloud storage"],
    "docker": ["docker", "containerization", "containers", "dockerfile"],
    "kubernetes": ["kubernetes", "k8s", "kube", "helm"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "oracle sql", "sql server", "mssql"],
    "nosql": ["nosql", "mongodb", "cassandra", "redis", "dynamodb", "elasticsearch"],
    "git": ["git", "github", "gitlab", "bitbucket", "version control"],
    "ci_cd": ["ci/cd", "jenkins", "github actions", "gitlab ci", "azure pipelines", "circleci", "travis"],
    "agile": ["agile", "scrum", "kanban", "sprint", "jira"],
}

# Long-form degree names and unambiguous abbreviations only.
EDUCATION_KEYWORDS: Dict[str, List[str]] = {
    "phd": ["phd", "ph.d", "doctorate", "doctoral"],
    "masters": ["master's degree", "masters degree", "m.sc", "m.tech", "mba", "mca"],
    "bachelors": ["bachelor", "bachelors", "b.sc", "b.tech", "bca", "b.com"],
    "diploma": ["diploma"],
}

CERTIFICATION_PATTERNS: List[str] = [
    r"\bAWS Certified:?\s+[A-Za-z0-9][A-Za-z0-9 +\-]{2,55}",
    r"\bMicrosoft Certified:?\s+[A-Za-z0-9][A-Za-z0-9 +\-]{2,55}",
    r"\bGoogle Cloud Certified:?\s+[A-Za-z0-9][A-Za-z0-9 +\-]{2,55}",
    r"\b(?:PMP|CISSP|CEH|CCNA|CCNP|OCI|CKA|CKAD)\b",
]


def normalize_keyword(keyword: str) -> str:
    return keyword.lower().strip()


def map_keyword_to_group(keyword: str) -> str:
    keyword_norm = normalize_keyword(keyword)
    for group_name, synonyms in SKILL_GROUPS.items():
        if keyword_norm == group_name or keyword_norm in [normalize_keyword(s) for s in synonyms]:
            return group_name
    return keyword_norm


def expand_keywords_with_synonyms(keywords: List[str]) -> Set[str]:
    expanded = set()
    for keyword in keywords:
        group = map_keyword_to_group(keyword)
        expanded.add(group)
        expanded.update(normalize_keyword(alias) for alias in SKILL_GROUPS.get(group, []))
    return expanded


def get_skill_groups() -> Dict[str, List[str]]:
    return SKILL_GROUPS.copy()


def get_education_keywords() -> Dict[str, List[str]]:
    return EDUCATION_KEYWORDS.copy()
